"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import clip

from .visual_grid_cliptag_embed import VisualGridClipTagEmbedding, image_text_simiarlity
from xmodaler.utils.initialization import trunc_normal_
from xmodaler.functional import noise_injection
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import EMBEDDING_REGISTRY

__all__ = ["VisualGridClipTagLREmbedding"]


def patch_text_simiarlity(
    patches_embeddings: torch.Tensor,
    texts_embeddings: torch.Tensor,
) -> torch.Tensor:
    patches_embeddings = patches_embeddings.float()                                       
    texts_embeddings = texts_embeddings.float()                                       
    patches_embeddings /= (patches_embeddings.norm(dim = -1, keepdim = True) + 1e-9)                         
    texts_embeddings /= (texts_embeddings.norm(dim = -1, keepdim = True) + 1e-9)                                   

    patch_sim = torch.einsum('bid,btd->bit', patches_embeddings, texts_embeddings)

    return patch_sim


@EMBEDDING_REGISTRY.register()
class VisualGridClipTagLREmbedding(VisualGridClipTagEmbedding):
    @configurable
    def __init__(
        self,
        *,
        replace_type: str,
        clip_local_ln_model: str,
        patch_tag_topk,
        **kwargs
    ):
        super(VisualGridClipTagLREmbedding, self).__init__(**kwargs)
        self.replace_type = replace_type
        self.embeddings_token_type = kwargs.pop('embeddings_token_type', None)
        self.patch_tag_topk = patch_tag_topk
        
        # load CLIP CLS Head
        model, _ = clip.load(clip_local_ln_model, device='cuda')
        model = model.float()
        with torch.no_grad():
            self.ln_post = model.visual.ln_post
            self.proj = model.visual.proj
        # Freeze in training
        self.ln_post.weight.requires_grad = False
        self.ln_post.bias.requires_grad = False
        self.proj.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        embeddings_pos = nn.Parameter(
            torch.zeros(1, cfg.DATALOADER.MAX_FEAT_NUM, cfg.MODEL.VISUAL_EMBED.OUT_DIM))
        trunc_normal_(embeddings_pos, std=.02)
        ret['embeddings_pos'] = embeddings_pos

        if cfg.MODEL.VISUAL_EMBED.CLIPTAG.TYPE_VOCAB_SIZE > 0:
            embeddings_token_type = nn.Embedding(
                cfg.MODEL.VISUAL_EMBED.CLIPTAG.TYPE_VOCAB_SIZE, cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            ret['embeddings_token_type'] = embeddings_token_type

        ret.update({
            'replace_type': cfg.MODEL.VISUAL_EMBED.CLIPTAG.REPLACE_TYPE,
            'clip_local_ln_model': cfg.MODEL.VISUAL_EMBED.CLIPTAG.CLIP_MODEL,
            'patch_tag_topk': cfg.MODEL.VISUAL_EMBED.CLIPTAG.TOPK2,
        })
        return ret

    @torch.no_grad()
    def postprocess_clip_local(self, att_feats):
        att_feats = self.ln_post(att_feats)
        att_feats = att_feats @ self.proj
        return att_feats

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.ATT_FEATS]
        ret = {}

        with torch.no_grad():
            if self.t_embeddings is not None:
                temperature = 0.01 

                logits = image_text_simiarlity(self.tag_vocab_embed, images_features = batched_inputs[kfg.GLOBAL_FEATS], temperature=temperature)
                top_k_probs, top_k_indices = torch.topk(logits, k = self.tag_topk, dim = -1) # (num_images, top_k)

                tag_mask = top_k_probs > self.tag_thres
                tag_mask = tag_mask.to(dtype=next(self.parameters()).dtype)

                tag_embed = self.tag_vocab_embed[top_k_indices]
                tag_embed = tag_embed * tag_mask.unsqueeze(-1)

                local_embed = self.postprocess_clip_local(feats)
                patch_sim = patch_text_simiarlity(local_embed, tag_embed)
                
                # channel-wise weighted-sum
                sim1 = torch.nn.functional.softmax(patch_sim / temperature, dim=-1)
                v_aware_tag_embed = (sim1.unsqueeze(-1) * tag_embed.unsqueeze(1)).sum(-2)
                
                # determine replace_mask using patch_sim
                batch_size, map_size, tag_size = patch_sim.shape
                if self.replace_type == 'top':
                    sim2 = torch.nn.functional.softmax((patch_sim * tag_mask.unsqueeze(1)).view(batch_size, -1) / temperature, dim=-1)

                    assert self.patch_tag_topk > 0
                    if self.patch_tag_topk < 1:
                        patch_tag_topk = int(self.patch_tag_topk * map_size * tag_size)
                    else:
                        patch_tag_topk = int(self.patch_tag_topk)

                    _, sel_top_k_indices = torch.topk(sim2, k = patch_tag_topk, dim = -1)
                    replace_mask = sim2.new_zeros(batch_size, map_size*tag_size)
                    replace_mask = torch.scatter(replace_mask, dim=-1, index=sel_top_k_indices, value=1)
                    replace_mask = (replace_mask.view(batch_size, map_size, tag_size).sum(-1) > 0).unsqueeze(-1).float()
                
                elif self.replace_type == 'random':
                    assert self.patch_tag_topk > 0 and self.patch_tag_topk < 1 # NOTE: quick hack, use patch_tag_topk as selected ratio
                    replace_ratio = self.patch_tag_topk if self.training else 1.0
                    replace_mask = torch.rand((batch_size, map_size), device=patch_sim.device) < replace_ratio
                    replace_mask = replace_mask.unsqueeze(-1).float()

                else:
                    raise NotImplementedError

        ################# Forward the same as VisualGridEmbedding
        if self.replace_type == 'all':
            embeddings = self.t_embeddings(v_aware_tag_embed)
        else:
            embeddings = self.embeddings(feats)
            if self.t_embeddings is not None:
                v_aware_tag_embed = self.t_embeddings(v_aware_tag_embed)
                if self.training:
                    embeddings = embeddings * (1 - replace_mask) + v_aware_tag_embed * replace_mask
                else:
                    embeddings = torch.cat([embeddings, v_aware_tag_embed], dim=-2)
                    tag_mask = replace_mask.squeeze(-1)
                    tag_mask = tag_mask.to(dtype=next(self.parameters()).dtype)
                    tag_mask = tag_mask.unsqueeze(1).unsqueeze(2)
                    ext_tag_mask = (1.0 - tag_mask) * -10000.0

                    att_mask = torch.cat([batched_inputs[kfg.ATT_MASKS], tag_mask], dim=-1)
                    ext_att_mask = torch.cat([batched_inputs[kfg.EXT_ATT_MASKS], ext_tag_mask], dim=-1)
                    ret.update({
                        kfg.ATT_MASKS: att_mask,
                        kfg.EXT_ATT_MASKS: ext_att_mask
                    })

        if self.g_embeddings is not None:
            if self.training and self.noise_inj_var > 0.0:
                g_feats = noise_injection(batched_inputs[kfg.GLOBAL_FEATS], variance=self.noise_inj_var, device = embeddings.device)
            else:
                g_feats = batched_inputs[kfg.GLOBAL_FEATS]
                
            g_embeddings = self.g_embeddings(g_feats)
            g_embeddings = g_embeddings.view(embeddings.shape[0], -1, embeddings.shape[-1])
            embeddings = torch.cat([g_embeddings, embeddings], dim=1)

        if self.training:
            embeddings_pos = self.embeddings_pos
        else:
            embeddings_pos = torch.cat([self.embeddings_pos, self.embeddings_pos[:, 1:]], dim=-2)
        embeddings = embeddings + embeddings_pos

        if self.embeddings_token_type is not None:
            token_type_ids = replace_mask.squeeze(-1).long()

            if self.training:
                token_type_ids = torch.cat([
                    token_type_ids.new_zeros(token_type_ids.shape[0], 1),
                    token_type_ids
                ], dim=-1)
            else:
                token_type_ids = torch.cat([
                    token_type_ids.new_zeros(token_type_ids.shape[0], self.embeddings_pos.shape[1]),
                    token_type_ids
                ], dim=-1)

            embeddings_token_type = self.embeddings_token_type(token_type_ids)
            embeddings = embeddings + embeddings_token_type

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        ret.update({ kfg.ATT_FEATS: embeddings })
        return ret

