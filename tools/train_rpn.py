import os
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
import cv2
import torch
import torch.nn

register_coco_instances("bdd_soda_train", {}, "/root/autodl-tmp/soda/Annotations/soda_train.json", "/root/autodl-tmp/soda/Images")
register_coco_instances("bdd_soda_test", {}, "/root/autodl-tmp/soda/Annotations/soda_test.json", "/root/autodl-tmp/soda/Images")
# 定义自己的数据增强方法
transform_list = [
    T.Resize((1200, 1200)),  # 使用固定尺寸 1200x1200
    T.RandomFlip(prob=0.5)  # 0.5的概率进行随机翻转
]


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-Detection/rpn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = ("bdd_soda_train",)
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "siou"
    cfg.MODEL.RPN.LOSS_WEIGHT = 0.9
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER
    cfg.OUTPUT_DIR = "./result"
    cfg.merge_from_list(args.opts)

    # 设定分布式训练相关的配置
    cfg.SOLVER.IMS_PER_BATCH = 5 * torch.cuda.device_count()
    cfg.DATALOADER.NUM_WORKERS = 3 * torch.cuda.device_count() # 可根据实际情况调整

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

class CustomTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        # 使用你自定义的数据增强和DatasetMapper
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=transform_list))

def main(args):
    cfg = setup(args)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


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