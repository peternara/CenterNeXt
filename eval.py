from utils.seed import setup_seed
from utils.parser import parse_yaml
from utils.data.dataset import DetectionDataset, collate_fn
from models.centernet import CenterNet, align_bboxes

import os
import numpy as np
import argparse
import torch
import cv2

from pathlib import Path
from tqdm import tqdm

from metric.pascalvoc import evaluate

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default="", help="Load weights to resume training")

    parser.add_argument("--dataset", type=str, default="./configs/datasets/voc0712.yaml")
    parser.add_argument("--model", type=str, default="./configs/models/r18_s4.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    
    parser.add_argument("--num-workers", default=4, type=int, help="Number of workers used in dataloading")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    return parser.parse_args()

@torch.inference_mode()
def evaluation(args, option, model=None, data_loader_val=None):
    # Load model
    if model is None:
        model = CenterNet(option).to(args.device)
        if os.path.isfile(args.weights):
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load dataset
    if data_loader_val is None:
        dataset_val = DetectionDataset(option, split="val")

        data_loader_val = torch.utils.data.DataLoader(dataset_val, 
                                                    option["OPTIMIZER"]["FORWARD_BATCHSIZE"] if "OPTIMIZER" in option else args.batch_size,
                                                    num_workers=args.num_workers,
                                                    shuffle=False,
                                                    pin_memory=False,
                                                    drop_last=False,
                                                    collate_fn=collate_fn)
        
    model.eval()
    with tqdm(data_loader_val, unit="batch") as tbar:
        for batch_data in tbar:
            batch_img = batch_data["img"].to(args.device)
            batch_org_img_shape = batch_data["org_img_shape"]
            batch_padded_ltrb = batch_data["padded_ltrb"]
            
            batch_output = model(batch_img)
            batch_output = model.post_processing(batch_output, batch_org_img_shape, batch_padded_ltrb, confidence_threshold=1e-2)
            
            for i in range(len(batch_img)):
                img_path = batch_data["img_path"][i]
                pred_bboxes = batch_output[i].cpu().numpy()
                
                #create detection file for evaluation
                detection_file = os.path.join("pred", Path(img_path).stem + ".txt")
                with open(detection_file, 'w') as f:
                    # img = cv2.imread(img_path)
                    for bbox in pred_bboxes:
                        class_id = int(bbox[0])
                        class_name = option["MODEL"]["CLASSES"][class_id]
                        xmin, ymin, xmax, ymax, conf = bbox[1:]
                        
                        f.write(f"{class_name} {conf} {xmin} {ymin} {xmax} {ymax}\n")
                    #     if conf > .1:
                    #         cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 5)
                    # cv2.imshow("img", img)
                    # cv2.waitKey(0)
                    
    mAP = evaluate(gtFolder="./gt",
                   gtFormat="xyrb",
                   gtCoordType="abs",
                   detFolder="./pred",
                   detFormat="xyrb",
                   detCoordType="abs")
    
    return mAP
    
if __name__ == "__main__":
    args = parse_args()
    os.makedirs("pred", exist_ok=True)
    
    # Parse yaml files
    dataset_option = parse_yaml(args.dataset)
    model_option = parse_yaml(args.model)
    collated_option = {**dataset_option, **model_option}
    
    mAP = evaluation(args, collated_option)
    print(f"mAP: {mAP}%")