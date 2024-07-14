"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn
import pickle

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import EMBEDDING_REGISTRY
from xmodaler.utils.initialization import trunc_normal_
from xmodaler.functional import noise_injection
from .visual_grid_embed import VisualGridEmbedding

__all__ = ["VisualGridClipTagEmbedding"]


def image_text_simiarlity(
    texts_embeddings: torch.Tensor,
    images_features: torch.Tensor,
    temperature: float = 0.01,
) -> torch.Tensor:
    # computing on cpu to avoid out of memory
    images_features = images_features.float()                                          # (num_images, clip_hidden_size)
    texts_embeddings = texts_embeddings.float()                                        # (num_categories, clip_hidden_size)
    images_features /= images_features.norm(dim = -1, keepdim = True)                                        # (num_images, clip_hidden_size)
    texts_embeddings /= texts_embeddings.norm(dim = -1, keepdim = True)                                      # (num_categories, clip_hidden_size)

    image_to_text_similarity = torch.matmul(images_features, texts_embeddings.transpose(1, 0)) / temperature # (num_imegs, num_categories)
    image_to_text_logits = torch.nn.functional.softmax(image_to_text_similarity, dim = -1)                   # (num_imegs, num_categories)
    
    return image_to_text_logits


@EMBEDDING_REGISTRY.register()
class VisualGridClipTagEmbedding(VisualGridEmbedding):
    @configurable
    def __init__(
        self, 
        *,
        in_dim: int,
        g_in_dim: int,
        out_dim: int,

        tag_vocab,
        tag_vocab_embed,
        tag_topk,
        tag_thres,

        **kwargs
    ):
        super(VisualGridClipTagEmbedding, self).__init__(
            in_dim = in_dim,
            g_in_dim = g_in_dim,
            out_dim = out_dim,
            **kwargs
        )

        self.t_embeddings = nn.Linear(g_in_dim, out_dim) if g_in_dim > 0 else None
        self.tag_vocab = tag_vocab
        self.tag_topk = tag_topk
        self.tag_thres = tag_thres
        self.register_buffer('tag_vocab_embed', tag_vocab_embed.float(), persistent=False) # False for cross-domain inference

    @classmethod
    def from_config(cls, cfg):
        kwargs = super().from_config(cfg)

        def _load_pkl_file(filepath):
            return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None

        tag_vocab, tag_vocab_embed = _load_pkl_file(cfg.MODEL.VISUAL_EMBED.CLIPTAG.VOCAB_EMBED_PATH)
        tag_vocab_embed = tag_vocab_embed.float()

        kwargs.update({
            'tag_vocab': tag_vocab,
            'tag_vocab_embed': tag_vocab_embed,
            'tag_topk': cfg.MODEL.VISUAL_EMBED.CLIPTAG.TOPK,
            'tag_thres': cfg.MODEL.VISUAL_EMBED.CLIPTAG.THRES,
        })

        embeddings_pos = nn.Parameter(
            torch.zeros(1, cfg.DATALOADER.MAX_FEAT_NUM+cfg.MODEL.VISUAL_EMBED.CLIPTAG.TOPK, cfg.MODEL.VISUAL_EMBED.OUT_DIM))
        trunc_normal_(embeddings_pos, std=.02)
        kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.ATT_FEATS]
        embeddings = self.embeddings(feats)
        ret = {}

        if self.t_embeddings is not None:
            logits = image_text_simiarlity(self.tag_vocab_embed, images_features = batched_inputs[kfg.GLOBAL_FEATS])
            top_k_probs, top_k_indices = torch.topk(logits, k = self.tag_topk, dim = -1) # (num_images, top_k)

            tag_embed = self.tag_vocab_embed[top_k_indices]
            tag_embed = self.t_embeddings(tag_embed)
            tag_embed = tag_embed.view(tag_embed.shape[0], -1, embeddings.shape[-1])
            embeddings = torch.cat([embeddings, tag_embed], dim=1)

            tag_mask = top_k_probs > self.tag_thres
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

        embeddings_pos = self.embeddings_pos
        embeddings = embeddings + embeddings_pos

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        ret.update({ kfg.ATT_FEATS: embeddings })
        return ret