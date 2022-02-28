# CenterNeXt
The official implementation of "CenterNeXt: Revisiting CenterNet in 2022"

## Results and Pre-trained Models

### VOC2007
| backbone | resolution | mAP | model config | weights |
|:---|:---:|:---:|:---:| :---:|
| ResNet-18 | 512 x 512  | 74.81 | [config](./configs/models/r18_s4.yaml) |[model]() |
| + Coupled head | 512 x 512  | 74.50 | [config](./configs/models/r18_s4_coupled.yaml) |[model]() |
| + Detection on lower resolution | 512 x 512  |  73.57  | [config](/configs/models/r18_s8_coupled.yaml) |[model]() |
| + Mosaic augmentation  | 512 x 512  | | [config](/configs/models/r18_s8_coupled_mosaic.yaml) |[model]() |
| + Mixup augmentation | 512 x 512  | | [config](/configs/models/r18_s8_coupled_mosaic_mixup.yaml) |[model]() |
| ResNet-50 | 512 x 512  | | [config](/configs/models/r50.yaml) |[model]() |
| ResNet-101 | 512 x 512  | | [config](/configs/models/r101.yaml) |[model]() |
| ConvNeXt-T | 512 x 512  | 83.30 | [config](/configs/models/convnext-t.yaml) |[model]() |

## Setup
Create a new conda virtual environment

```
conda create -n centernext python=3.8 -y
conda activate centernext
```

Install Pytorch and torchvision following official instructions. For example:

```
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Clone this repo and install required packages:
```
git clone https://github.com/MarkAny-Vision-AI/CenterNeXt
cd CenterNeXt
pip install -r requirements.txt
```

Download a dataset:

if (your_os == 'Window'):
```
cd CenterNeXt
scripts/download-voc0712.bat
python create_gt_files.py
```
else:
```
cd CenterNeXt
scripts/download-voc0712.sh
python create_gt_files.py
```

### Training
```
python train.py --model ./configs/models/your_model.yaml
```

### Evaluation
```
python eval.py --model ./configs/models/your_model.yaml --weigths /path/to/your_model.pth
```
