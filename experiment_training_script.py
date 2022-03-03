import os
import glob
import torch
from pathlib import Path

os.system("python create_gt_files.py")

models_cfg = glob.glob("./configs/models/*.yaml")
num_gpus = torch.cuda.device_count()
print(f"num_gpus: {num_gpus}")

for model_cfg in models_cfg:
    if num_gpus == 1:
        cmd = f"python train.py --model {model_cfg} --save-folder {Path(model_cfg).stem}"
    else:
        cmd = f"torchrun --nproc_per_node={num_gpus} train.py --model {model_cfg} --save-folder {Path(model_cfg).stem}"
    print(cmd)
    os.system(cmd)
