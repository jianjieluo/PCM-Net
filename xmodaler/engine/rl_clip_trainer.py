# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
import random
from .defaults import DefaultTrainer
from xmodaler.scorer import build_scorer
from xmodaler.config import kfg
from xmodaler.losses import build_rl_losses
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY

__all__ = ['RLClipTrainer']

@ENGINE_REGISTRY.register()
class RLClipTrainer(DefaultTrainer):
    """
    Use sample max in training sampling    
    """

    def __init__(self, cfg):
        super(RLClipTrainer, self).__init__(cfg)
        self.clip_scorer = self.build_scorer(cfg)
        self.losses = build_rl_losses(cfg)

    @classmethod
    def build_scorer(cls, cfg):
        return build_scorer(cfg)

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"

        start = time.perf_counter()
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter//self.iters_per_epoch)

            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)
        data_time = time.perf_counter() - start

        data = comm.unwrap_model(self.model).preprocess_batch(data)
        output_semantic = (kfg.DEMO_STR in data)

        self.model.eval()
        with torch.no_grad():
            bs_data = copy.copy(data)
            bs_outputs_dict = self.model(bs_data, use_beam_search=False, output_sents=True, output_semantic=output_semantic)
            # CLIPScore baseline
            bs_clip_rewards = self.clip_scorer(bs_outputs_dict)

        self.model.train()
        data[kfg.DECODE_BY_SAMPLE] = True
        outputs_dict = self.model(data, use_beam_search=False, output_sents=True, output_semantic=output_semantic)

        clip_rewards = self.clip_scorer(outputs_dict)
        clip_rewards = torch.from_numpy(clip_rewards[kfg.REWARDS] - bs_clip_rewards[kfg.REWARDS]).float().cuda()
        outputs_dict.update({ kfg.REWARDS: clip_rewards })

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)
        
        losses = [losses_dict[k] for k in losses_dict if 'acc' not in k]
        losses = sum(losses)

        self.optimizer.zero_grad()
        losses.backward()

        bs_clip_rewards.pop(kfg.REWARDS)
        losses_dict.update(bs_clip_rewards)

        self._write_metrics(losses_dict, data_time)
        self.optimizer.step()