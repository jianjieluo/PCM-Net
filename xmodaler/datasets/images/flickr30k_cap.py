"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import os

from .mscoco import MSCoCoDataset
from xmodaler.config import configurable, kfg
from ..build import DATASETS_REGISTRY

__all__ = ["Flickr30kCapDataset"]

@DATASETS_REGISTRY.register()
class Flickr30kCapDataset(MSCoCoDataset):

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = super().from_config(cfg, stage)

        # update anno files name
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "flickr30k_caption_anno_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "flickr30k_caption_anno_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "flickr30k_caption_anno_test.pkl")
        }
        ret.update(
            { "anno_file": ann_files[stage] }
        ) 
        return ret

