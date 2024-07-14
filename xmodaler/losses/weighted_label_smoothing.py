"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class WeightedLabelSmoothing(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        label_smoothing,
        alpha,
        gamma,
        gamma_type
    ):
        super(WeightedLabelSmoothing, self).__init__()
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - self.label_smoothing
        self.criterion = nn.KLDivLoss(reduction='none')

        self.w = 2.5 # NOTE: quick hack, the preprocessed `clipscore` in the dataloader have been multipled 2.5
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_type = gamma_type


    @classmethod
    def from_config(cls, cfg):
        return {
            "label_smoothing": cfg.LOSSES.LABELSMOOTHING,
            "alpha": cfg.LOSSES.ALPHA,
            "gamma": cfg.LOSSES.GAMMA,
            "gamma_type": cfg.LOSSES.GAMMA_TYPE
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def Forward(self, logits, targets, loss_weights):
        bs, seq_len, _ = logits.shape
        loss_weights = loss_weights.unsqueeze(-1).expand(bs, seq_len).reshape(-1)

        logP = F.log_softmax(logits.view(-1, logits.shape[-1]), dim=-1) 
        targets = targets.view(-1)
        mask = targets >= 0

        assign_seq = targets  #.type(torch.cuda.LongTensor)
        assign_seq[assign_seq < 0] = 0

        if self.gamma > 0:
            probs = F.softmax(logits.view(-1, logits.shape[-1]), dim=-1) 
            selected_probs = torch.gather(probs, index=assign_seq.unsqueeze(-1), dim=1)
            gamma_weights = (1 - selected_probs) ** self.gamma

        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss = self.criterion(logP, true_dist).sum(1)
        loss = loss * loss_weights
        if self.gamma > 0:

            if self.gamma_type == 'word':
                gamma_weights = gamma_weights.squeeze(-1)
                loss = loss * gamma_weights
            elif self.gamma_type == 'batch': # NOTE: we found this can achieve the best performance
                loss = loss * gamma_weights
                total_word_num = mask.shape[0]
                mask = mask.unsqueeze(0).expand(total_word_num, -1) * mask.unsqueeze(-1)
            elif self.gamma_type == 'sent':
                loss = loss * gamma_weights
                total_word_num = mask.shape[0]
                mask1 = mask.unsqueeze(0).expand(total_word_num, -1) * mask.unsqueeze(-1)

                mask2 = torch.eye(bs, device=mask1.device).view(-1, 1).expand(-1, seq_len).reshape(bs, -1)
                mask2 = mask2.unsqueeze(1).expand(bs, seq_len, -1).reshape(bs*seq_len, -1)
                mask = (mask1 * mask2).bool()
            else:
                raise NotImplementedError

        loss = torch.masked_select(loss, mask).mean()
        return loss

    def forward(self, outputs_dict):
        loss_weights = outputs_dict[kfg.CLIP_SCORE]
        loss_weights = (loss_weights / self.w) * self.alpha
        loss_weights = torch.clamp(loss_weights, min=0.0, max=1.0)

        ret  = {}
        if kfg.G_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.G_LOGITS]
            targets = outputs_dict[kfg.G_TARGET_IDS]
            loss = self.Forward(logits, targets, loss_weights)
            ret.update({ 'LabelSmoothing(G) loss': loss })

        if kfg.U_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.U_LOGITS]
            targets = outputs_dict[kfg.U_TARGET_IDS]
            loss = self.Forward(logits, targets, loss_weights)
            ret.update({ 'LabelSmoothing(U) loss': loss })
        return ret