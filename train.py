from utils.seed import setup_seed
from utils.parser import parse_yaml
from utils.data.dataset import DetectionDataset, collate_fn
from models.centernet import CenterNet
from eval import evaluation

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default="", help="Load weights to resume training")

    parser.add_argument("--dataset", type=str, default="./configs/datasets/voc0712.yaml")
    parser.add_argument("--model", type=str, default="./configs/models/r18_s4.yaml")
    parser.add_argument("--optimizer", type=str, default="./configs/optimizers/base_optimizer.yaml")
    
    parser.add_argument("--num-workers", default=4, type=int, help="Number of workers used in dataloading")
    parser.add_argument("--save-folder", default="./weights", type=str, help="Where you save weights")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument("--seed", default=7777, type=int)
    return parser.parse_args()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train(args, option):
    # Load model
    model = CenterNet(option).to(args.device)
        
    # Load dataset
    dataset_train = DetectionDataset(option, split="train", apply_augmentation=True)
    dataset_val = DetectionDataset(option, split="val")
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, 
                                                    option["OPTIMIZER"]["FORWARD_BATCHSIZE"],
                                                    num_workers=args.num_workers,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    drop_last=True,
                                                    collate_fn=collate_fn)

    data_loader_val = torch.utils.data.DataLoader(dataset_val, 
                                                  option["OPTIMIZER"]["FORWARD_BATCHSIZE"],
                                                  num_workers=args.num_workers,
                                                  shuffle=False,
                                                  pin_memory=False,
                                                  drop_last=False,
                                                  collate_fn=collate_fn)
    
    # Load optimizer & lr scheduler
    optimizer = optim.Adam(model.parameters(), 
                            lr=option["OPTIMIZER"]["LR"],
                            weight_decay=option["OPTIMIZER"]["WD"])
    
    lr_scheduler = create_lr_scheduler_with_warmup(optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                                        T_max=len(data_loader_train) * option["OPTIMIZER"]["EPOCHS"]),
                            warmup_start_value=1e-10,
                            warmup_end_value=option["OPTIMIZER"]["LR"],
                            warmup_duration=option["OPTIMIZER"]["WARMUP_ITERATIONS"]
                            )
    
    # Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Logger
    logger = SummaryWriter()
    
    start_epoch = 1
    total_epoch = option["OPTIMIZER"]["EPOCHS"]
    
    iters_to_accumulate = max(round(option["OPTIMIZER"]["STEP_BATCHSIZE"]/option["OPTIMIZER"]["FORWARD_BATCHSIZE"]), 1)

    best_mAP = 0.
    
    # Load Pretrained Weights
    if os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights)
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint['mAP']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    for epoch in range(start_epoch, total_epoch + 1):
        # Training
        model.train()
        with tqdm(data_loader_train, unit="batch") as tbar:
            for idx_batch, batch_data in enumerate(tbar):
                # Update lr_scheduler
                lr_scheduler(None)
                
                n_iter = idx_batch + (epoch - 1) * len(data_loader_train)
                tbar.set_description(f"Epoch {epoch}/{total_epoch}")
                
                batch_img = batch_data["img"].to(args.device)
                batch_label = batch_data["label"]
                
                #Forward
                with torch.cuda.amp.autocast():
                    batch_output = model(batch_img)
                    loss, losses = model.compute_loss(batch_output, batch_label)
                
                #Plot
                tbar.set_postfix(loss=loss.item())
                logger.add_scalar('train/loss_offset_xy', losses[0].item(), n_iter)
                logger.add_scalar('train/loss_wh', losses[1].item(), n_iter)
                logger.add_scalar('train/loss_class_heatmap', losses[2].item(), n_iter)
                logger.add_scalar('train/loss', loss.item(), n_iter)
                logger.add_scalar('train/lr', get_lr(optimizer), n_iter)

                #Backword
                loss = loss / iters_to_accumulate
                scaler.scale(loss).backward()

                if (n_iter + 1) % iters_to_accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
        # Validation
        model.eval()
        mAP = evaluation(args, option, model, data_loader_val) # mAP
        print(f"mAP: {mAP}%")
        logger.add_scalar('val/mAP', mAP, epoch)
        
        # Save weights
        checkpoint = {
        'epoch': epoch,# zero indexing
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'lr_scheduler_state_dict' : lr_scheduler.state_dict(),
        'scaler_state_dict' : scaler.state_dict(),
        'mAP': mAP,
        }
        
        if best_mAP < mAP:
            best_mAP = mAP
            torch.save(checkpoint, os.path.join(args.save_folder, 'best_mAP.pth'))
        torch.save(checkpoint, os.path.join(args.save_folder, 'epoch_' + str(epoch) + '.pth'))       

if __name__ == "__main__":
    args = parse_args()
    
    setup_seed(args.seed)
    os.makedirs(args.save_folder, exist_ok=True)
    
    # Parse yaml files
    dataset_option = parse_yaml(args.dataset)
    model_option = parse_yaml(args.model)
    optimizer_option = parse_yaml(args.optimizer)
    collated_option = {**dataset_option, **model_option, **optimizer_option}
    
    train(args, collated_option)