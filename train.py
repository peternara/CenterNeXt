from utils.seed import setup_seed
from utils.parser import parse_yaml
from utils.data.dataset import DetectionDataset, collate_fn
import utils.ddp
from models.centernet import CenterNet, compute_loss
from eval import evaluation_on_voc, evaluation_on_coco

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

    # distributed training parameters
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train(args, option):
    utils.ddp.init_distributed_mode(args)
    
    print(args)

    device = torch.device(args.device)
    
    # Fix the seed for reproducibility
    seed = args.seed + utils.ddp.get_rank()
    setup_seed(seed)

    # Load model
    model = CenterNet(option).to(args.device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # Load dataset
    dataset_train = DetectionDataset(option, split="train", apply_augmentation=True)
    dataset_val = DetectionDataset(option, split="val")
    
    assert option["OPTIMIZER"]["STEP_BATCHSIZE"] >= option["OPTIMIZER"]["FORWARD_BATCHSIZE"]
    assert option["OPTIMIZER"]["STEP_BATCHSIZE"] % option["OPTIMIZER"]["FORWARD_BATCHSIZE"] == 0
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=utils.ddp.get_world_size(), rank=utils.ddp.get_rank(), shuffle=True
            )
        
        assert option["OPTIMIZER"]["STEP_BATCHSIZE"] % utils.ddp.get_world_size() == 0
        assert option["OPTIMIZER"]["FORWARD_BATCHSIZE"] % utils.ddp.get_world_size() == 0
        
        option["OPTIMIZER"]["STEP_BATCHSIZE"] = option["OPTIMIZER"]["STEP_BATCHSIZE"] // utils.ddp.get_world_size()
        option["OPTIMIZER"]["FORWARD_BATCHSIZE"] = option["OPTIMIZER"]["FORWARD_BATCHSIZE"] // utils.ddp.get_world_size()
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        
    data_loader_train = torch.utils.data.DataLoader(dataset_train, 
                                                    option["OPTIMIZER"]["FORWARD_BATCHSIZE"],
                                                    num_workers=args.num_workers,
                                                    sampler=sampler_train,
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
    optimizer = optim.Adam(model_without_ddp.parameters(), 
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
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        # Training
        model.train()
        with tqdm(data_loader_train, unit="batch", disable=not utils.ddp.is_main_process()) as tbar:
            for idx_batch, batch_data in enumerate(tbar):
                # Update lr_scheduler
                lr_scheduler(None)
                
                n_iter = idx_batch + (epoch - 1) * len(data_loader_train)
                
                batch_img = batch_data["img"].to(args.device)
                batch_label = batch_data["label"]
                
                #Forward
                with torch.cuda.amp.autocast():
                    batch_output = model(batch_img)
                    loss, losses = compute_loss(batch_output, batch_label)
                
                #Plot
                if utils.ddp.is_main_process():
                    tbar.set_description(f"Epoch {epoch}/{total_epoch}")
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
        
        if utils.ddp.is_main_process():
            # Validation
            model_without_ddp.eval()
            
            # SHOULD I STUDY DESIGN PATTERN?
            if collated_option["DATASET"]["NAME"] == "VOC0712":
                mAP = evaluation_on_voc(args, option, model_without_ddp, data_loader_val)
            elif collated_option["DATASET"]["NAME"] == "COCO":
                mAP = evaluation_on_coco(args, option, model_without_ddp, data_loader_val)
            print(f"mAP: {mAP}%")
            logger.add_scalar('val/mAP', mAP, epoch)
            
            # Save weights
            checkpoint = {
            'epoch': epoch,# zero indexing
            'model_state_dict': model_without_ddp.state_dict(),
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
    
    os.makedirs(args.save_folder, exist_ok=True)
    
    # Parse yaml files
    dataset_option = parse_yaml(args.dataset)
    model_option = parse_yaml(args.model)
    optimizer_option = parse_yaml(args.optimizer)
    collated_option = {**dataset_option, **model_option, **optimizer_option}
    
    train(args, collated_option)
