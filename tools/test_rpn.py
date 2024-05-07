import os
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer,default_argument_parser, launch
import cv2
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from copy import deepcopy
import torch
import numpy as np
import random
import os

register_coco_instances("bdd_soda_train", {}, "/root/autodl-tmp/soda/Annotations/soda_train.json", "/root/autodl-tmp/soda/Images")
register_coco_instances("bdd_soda_test", {}, "/root/autodl-tmp/soda/Annotations/soda_test.json", "/root/autodl-tmp/soda/Images")

# 定义自己的数据增强方法
transform_list = [
    T.Resize((1200, 1200)),  # 使用固定尺寸 1200x1200
]

# 1. 配置
def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-Detection/rpn_R_50_FPN_1x.yaml")
    cfg.MODEL.WEIGHTS = "/root/autodl-tmp/RegionCLIP/pretrained_ckpt/rpn/soda_6_cls_siouRPN.pth"  # 替换为您的模型权重文件路径
    cfg.DATASETS.TRAIN = ("bdd_soda_train",)
    cfg.DATASETS.TEST = ("bdd_soda_test",)
    cfg.MODEL.RPN.NMS_THRESH = 0.5
    cfg.DATALOADER.NUM_WORKERS = 3 * torch.cuda.device_count()

    return cfg

class CustomTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        # 使用你自定义的数据增强和DatasetMapper
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=transform_list))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name=None):
        return build_detection_test_loader(cfg, dataset_name,mapper=DatasetMapper(cfg, is_train=False, augmentations=transform_list))

# 2. 创建评估器
evaluator = COCOEvaluator("bdd_soda_test", False, output_dir="../output/")

# 3. 进行评估
def main(args):
    cfg = setup(args)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True)  # 加载训练好的权重
    trainer.test(cfg, trainer.model, evaluators=[evaluator])

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )