import os
import glob
from pathlib import Path

os.system("python create_gt_files.py")

models_cfg = glob.glob("./configs/models/*.yaml")
for model_cfg in models_cfg:
    os.system(f"python train.py --model {model_cfg} --save-folder {Path(model_cfg).stem}")
