from .dcn import *

import numpy as np
import math
import torch
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn
import timm

import torchvision.transforms.functional as F

def align_bboxes(normalized_bboxes, resized_img_shape, padded_ltrb, org_img_shape):
    normalized_bboxes[:, [1, 3]] *= resized_img_shape[0]
    normalized_bboxes[:, [2, 4]] *= resized_img_shape[1]
                        
    normalized_bboxes[:, [1, 3]] -= padded_ltrb[0]
    normalized_bboxes[:, [2, 4]] -= padded_ltrb[1]
    
    non_padded_img_shape = [resized_img_shape[0] - padded_ltrb[0] - padded_ltrb[2], 
                            resized_img_shape[1] - padded_ltrb[1] - padded_ltrb[3]]
    
    normalized_bboxes[:, [1, 3]] /= non_padded_img_shape[0]
    normalized_bboxes[:, [2, 4]] /= non_padded_img_shape[1]
    
    normalized_bboxes[:, [1, 3]] *= org_img_shape[0]
    normalized_bboxes[:, [2, 4]] *= org_img_shape[1]
    
    normalized_bboxes[:, [1, 3]] = torch.clamp(normalized_bboxes[:, [1, 3]], 0, org_img_shape[0])
    normalized_bboxes[:, [2, 4]] = torch.clamp(normalized_bboxes[:, [2, 4]], 0, org_img_shape[1])
    return normalized_bboxes

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[:, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    
class Upsamling(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=2):
        super(Upsamling, self).__init__()
        # deconv basic config
        if ksize == 4:
            padding = 1
            output_padding = 0
        elif ksize == 3:
            padding = 1
            output_padding = 1
        elif ksize == 2:
            padding = 0
            output_padding = 0
        
        self.conv = DeformableConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.up = nn.ConvTranspose2d(out_channels, out_channels, ksize, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        fill_up_weights(self.up)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv(x)))
        x = torch.relu(self.bn2(self.up(x)))
        return x

class CenterNet(nn.Module):
    def __init__(self, option):
        super(CenterNet, self).__init__()
        
        self.num_classes = len(option["MODEL"]["CLASSES"])
        self.stride = option["MODEL"]["STRIDE"]
        self.coupled_localization_branch = option["MODEL"]["COUPLED_LOCALIZATION_BRANCH"] 
        
        self.backbone = timm.create_model(model_name=option["MODEL"]["BACKBONE"],
                                          pretrained=option["MODEL"]["USE_PRETRAINED_BACKBONE"],
                                          features_only=True)
        
        self.stage_channels = self.backbone.feature_info.channels()
        
        self.upsample1 = Upsamling(self.stage_channels[-1], 256, ksize=4, stride=2) # 32 -> 16
        self.upsample2 = Upsamling(256, 128, ksize=4, stride=2) # 16 -> 8
        self.upsample3 = Upsamling(128, 64, ksize=4, stride=2) if self.stride == 4 else nn.Identity()  #  8 -> 4
        
        head_dim = 64 if self.stride == 4 else 128

        self.cls_pred = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, self.num_classes, kernel_size=1)
        )
        
        init_prob = 0.01
        nn.init.constant_(self.cls_pred[-1].bias, -torch.log(torch.tensor((1.-init_prob)/init_prob)))
        
        if self.coupled_localization_branch == False:
            self.txty_pred = nn.Sequential(
                nn.Conv2d(head_dim, head_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_dim, 2, kernel_size=1)
            )
            self.twth_pred = nn.Sequential(
                nn.Conv2d(head_dim, head_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_dim, 2, kernel_size=1)
            )
        else:
            self.txtytwth_pred = nn.Sequential(
                nn.Conv2d(head_dim, head_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_dim, 4, kernel_size=1)
            )
        
        #for decoding
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.max_num_dets = 100

    def encode(self, x):
        x = self.backbone(x)
        x = x[-1]
        
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        
        cls_pred = self.cls_pred(x)
        
        if self.coupled_localization_branch == False:
            txty_pred = self.txty_pred(x)
            twth_pred = self.twth_pred(x)
            out = torch.cat([txty_pred, twth_pred, cls_pred], dim=1)
        else:
            twtytwth_pred = self.txtytwth_pred(x)
            out = torch.cat([twtytwth_pred, cls_pred], dim=1)
        
        return out
    
    def forward(self, x):
        self.img_h, self.img_w = x.shape[2:]

        out = self.encode(x)
        
        if self.training:
            return out
        else:
            out_h, out_w = out.shape[2:]
            device = out.device
            
            grid_y = torch.arange(out_h, dtype=out.dtype, device=device).view(1, out_h, 1).repeat(1, 1, out_w)
            grid_x = torch.arange(out_w, dtype=out.dtype, device=device).view(1, 1, out_w).repeat(1, out_h, 1)
            
            # localization
            bboxes_cx = (self.stride * (grid_x + out[:, 0]).flatten(start_dim=1))/self.img_w
            bboxes_cy = (self.stride * (grid_y + out[:, 1]).flatten(start_dim=1))/self.img_h
            
            bboxes_w = (self.stride * out[:, 2].flatten(start_dim=1))/self.img_w
            bboxes_h = (self.stride * out[:, 3].flatten(start_dim=1))/self.img_h
            
            bboxes_xmin = bboxes_cx - bboxes_w/2
            bboxes_ymin = bboxes_cy - bboxes_h/2
            
            bboxes_xmax = bboxes_cx + bboxes_w/2
            bboxes_ymax = bboxes_cy + bboxes_h/2
            
            class_heatmap = torch.sigmoid(out[:, 4:])# [B, 20, H, W]
            class_heatmap = self.nms(class_heatmap).flatten(start_dim=2).transpose(1, 2) # [B, 20, H*W] -> # [B, H*W, 20]
            class_heatmap, class_idx = torch.max(class_heatmap, dim=2) # [B, H*W]
            class_idx = class_idx.to(dtype=out.dtype)
            
            _, topk_inds = torch.topk(class_heatmap, k=self.max_num_dets, dim=1)
            
            out = [torch.gather(x, dim=1, index=topk_inds) 
                   for x in [class_idx, bboxes_xmin, bboxes_ymin, bboxes_xmax, bboxes_ymax, class_heatmap]]
            out = torch.stack(out, dim=2) # [B, self.max_num_dets, 6]
            return out
    
    def nms(self, class_probability:Tensor) -> Tensor:
        mask = torch.eq(class_probability, self.max_pool(class_probability)).to(class_probability.dtype)
        return class_probability * mask
    
    def post_processing(self, 
                        batch_bboxes : Tensor,
                        batch_org_img_shape,
                        batch_padded_ltrb, 
                        confidence_threshold : float=1e-2):
        self.eval()
        with torch.no_grad():
            filtered_batch_bboxes = []
            for bboxes, org_img_shape, padded_ltrb in zip(batch_bboxes, batch_org_img_shape, batch_padded_ltrb):                
                bboxes_confidence = bboxes[:, 5]
                confidence_mask = bboxes_confidence > confidence_threshold
                filtered_bboxes = []
                
                if torch.count_nonzero(confidence_mask) > 0:    
                    filtered_bboxes = bboxes[confidence_mask]
                    filtered_bboxes = align_bboxes(normalized_bboxes=filtered_bboxes,
                                                resized_img_shape=(self.img_w, self.img_h),
                                                padded_ltrb=padded_ltrb,
                                                org_img_shape=org_img_shape)
                
                filtered_batch_bboxes.append(filtered_bboxes)
            return filtered_batch_bboxes
   
def compute_loss(batch_pred, batch_label):
    batch_size = batch_pred.shape[0]
    dtype = batch_pred.dtype
    device = batch_pred.device
    
    loss_offset_xy_function = nn.L1Loss(reduction='none')
    loss_wh_function = nn.L1Loss(reduction='none')

    batch_label["bboxes_regression"] = batch_label["bboxes_regression"].to(device)
    batch_label["classes_gaussian_heatmap"] = batch_label["classes_gaussian_heatmap"].to(device)
    batch_label["foreground"] = batch_label["foreground"].to(device)
    
    batch_loss_offset_x = torch.tensor(0., dtype=dtype, device=device)
    batch_loss_offset_y = torch.tensor(0., dtype=dtype, device=device)
    batch_loss_w = torch.tensor(0., dtype=dtype, device=device)
    batch_loss_h = torch.tensor(0., dtype=dtype, device=device)
    batch_loss_class_heatmap = torch.tensor(0., dtype=dtype, device=device)
    
    batch_loss_offset_x = loss_offset_xy_function(batch_pred[:, 0], batch_label["bboxes_regression"][:, 0]) * batch_label["foreground"]/batch_size
    batch_loss_offset_y = loss_offset_xy_function(batch_pred[:, 1], batch_label["bboxes_regression"][:, 1]) * batch_label["foreground"]/batch_size
    batch_loss_w = loss_wh_function(batch_pred[:, 2], batch_label["bboxes_regression"][:, 2]) * batch_label["foreground"]/batch_size
    batch_loss_h = loss_wh_function(batch_pred[:, 3], batch_label["bboxes_regression"][:, 3]) * batch_label["foreground"]/batch_size

    batch_loss_class_heatmap = focal_loss(batch_pred[:, 4:], batch_label["classes_gaussian_heatmap"])/batch_size

    batch_loss_offset_x = batch_loss_offset_x.flatten(1).sum(1)
    batch_loss_offset_y = batch_loss_offset_y.flatten(1).sum(1)
    batch_loss_w = batch_loss_w.flatten(1).sum(1)
    batch_loss_h = batch_loss_h.flatten(1).sum(1)
    batch_loss_class_heatmap = batch_loss_class_heatmap.flatten(1).sum(1)

    batch_num_positive_samples = batch_label["foreground"].flatten(1).sum(1)
    batch_num_positive_samples = torch.maximum(batch_num_positive_samples, torch.ones_like(batch_num_positive_samples)) # to avoid zero divide
    
    batch_loss_offset_x /= batch_num_positive_samples
    batch_loss_offset_y /= batch_num_positive_samples
    batch_loss_w /= batch_num_positive_samples
    batch_loss_h /= batch_num_positive_samples
    batch_loss_class_heatmap /= batch_num_positive_samples

    batch_loss_offset_xy = torch.sum(batch_loss_offset_x + batch_loss_offset_y)/2.
    batch_loss_wh = 0.1 * torch.sum(batch_loss_w + batch_loss_h)/2.
    batch_loss_class_heatmap = torch.sum(batch_loss_class_heatmap)
    loss = batch_loss_offset_xy + batch_loss_wh + batch_loss_class_heatmap
    return loss, [batch_loss_offset_xy, batch_loss_wh, batch_loss_class_heatmap]

#Read Training-Time-Friendly Network for Real-Time Object Detection paper for more details
def focal_loss(pred, gaussian_kernel, alpha=2., beta=4.):
    bce_loss_function = nn.BCEWithLogitsLoss(reduction='none')
        
    gt = (gaussian_kernel == 1.).float()
    
    positive_mask = gaussian_kernel == 1.
    negative_mask = ~positive_mask
    
    loss = bce_loss_function(pred, gt)
    
    if torch.count_nonzero(positive_mask) > 0:
        loss[positive_mask] *= (1. - pred[positive_mask].sigmoid()) ** alpha
        
    if torch.count_nonzero(negative_mask) > 0:
        loss[negative_mask] *= ((1. - gaussian_kernel[negative_mask]) ** beta) * (pred[negative_mask].sigmoid() ** alpha)
        
    return loss

