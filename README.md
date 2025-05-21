# Cascade Open Vocabulary Small Object Detection

<p align="center">
<img src="overview.png" width=97% height=97%
class="center">
</p>

## Installation
Our project builds on the foundation of RegionCLIP.
The installation follows [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md). Here we provide a quickstart guide, and refer to [the solutions from Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#common-installation-issues) should any issue arise.

#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.6 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
- OpenCV is optional for training and inference, yet is needed by our demo and visualization

```
cd COVD
python -m pip install -e RegionCLIP
```

Related weights file downloads:https://drive.google.com/drive/folders/1i1jAzfD2tAVwHNzd2qIdexYNEg1IUEhd?usp=drive_link
Some weights will be uploaded at a later date.

If you have any questions, please contact yulongroy@gmail.com