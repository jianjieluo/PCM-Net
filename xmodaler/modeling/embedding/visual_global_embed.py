import math
import torch
from torch import nn
from typing import Tuple, Optional, Union
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.utils.initialization import trunc_normal_
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY

__all__ = ["VisualGlobalEmbedding"]


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act='tanh'):
        super(MLP, self).__init__()
        if act == 'tanh':
            act = nn.Tanh
        elif act == 'gelu':
            act = nn.GELU
        elif act == 'relu':
            act = nn.ReLU
        else:
            act = nn.Tanh
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    
    def initialization(self, init_method: str = "normal", sigma: float = 0.1):
        def normal_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight.data, 0, sigma)
                nn.init.zeros_(m.bias.data)
        if init_method == "normal":
            self.model.apply(normal_init_weights)


@EMBEDDING_REGISTRY.register()
class VisualGlobalEmbedding(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        prompt_len: int,
        g_in_dim: int,
        out_dim: int,
        **kwargs
    ):
        super(VisualGlobalEmbedding, self).__init__()
        self.prompt_len = prompt_len
        self.embeddings = kwargs["embeddings"]
        self.g_embeddings = nn.Linear(g_in_dim, out_dim) if g_in_dim > 0 else None
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop('embeddings_pos', None)

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "prompt_len": cfg.MODEL.VISUAL_EMBED.PROMPT_LEN,
            "g_in_dim": cfg.MODEL.VISUAL_EMBED.G_IN_DIM,
            "out_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM
        }

        embeddings = MLP(
            (
                cfg.MODEL.VISUAL_EMBED.G_IN_DIM, 
                (cfg.MODEL.VISUAL_EMBED.OUT_DIM * cfg.MODEL.VISUAL_EMBED.PROMPT_LEN) // 2,
                cfg.MODEL.VISUAL_EMBED.OUT_DIM * cfg.MODEL.VISUAL_EMBED.PROMPT_LEN
            )
        )
        kwargs['embeddings'] = embeddings

        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.VISUAL_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.VISUAL_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.VISUAL_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.VISUAL_EMBED.USE_NORM:
            embeddings_norm = nn.LayerNorm(cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        embeddings_pos = nn.Parameter(
            torch.zeros(1, cfg.DATALOADER.MAX_FEAT_NUM, cfg.MODEL.VISUAL_EMBED.OUT_DIM))
        trunc_normal_(embeddings_pos, std=.02)
        kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.GLOBAL_FEATS]
        batch_size = feats.shape[0]
        embeddings = self.embeddings(feats)
        embeddings = embeddings.view(batch_size, self.prompt_len, -1)

        if self.g_embeddings is not None:
            g_embeddings = self.g_embeddings(batched_inputs[kfg.GLOBAL_FEATS])
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

        return { kfg.ATT_FEATS: embeddings }