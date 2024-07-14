import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg

from .build import ENCODER_REGISTRY, build_encoder

__all__ = ["MapperEncoder"]


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act='tanh'):
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


class Mapping_Network(nn.Module):
    def __init__(self, sizes, bias=True, act='tanh'):
        super(Mapping_Network, self).__init__()
        self.model = MLP(sizes, bias, act)

    def initialization(self, init_method: str = "normal", sigma: float = 0.1):
        self.model.initialization(init_method, sigma)

    def forward(self, x: torch.Tensor, training=True) -> torch.Tensor:
        if training:
            self.model.train()
            return x + self.model(x)
        else:
            self.model.eval()
            with torch.no_grad():
                return x + self.model(x)


@ENCODER_REGISTRY.register()
class MapperEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        prompt_len: int,
        pre_encoder,
        mapper,
    ):
        super(MapperEncoder, self).__init__()
        self.pre_encoder = pre_encoder
        self.mapper = mapper
        self.prompt_len = prompt_len

    @classmethod
    def from_config(cls, cfg):
        tmp_cfg = cfg.clone()
        tmp_cfg.defrost()
        tmp_cfg.MODEL.ENCODER = cfg.MODEL.CLIPCAP.PRE_ENCODER
        tmp_cfg.freeze()
        pre_encoder = build_encoder(tmp_cfg)

        if cfg.MODEL.CLIPCAP.MAPPER_TYPE == 'MLP':
            mapper = MLP(
                (
                    cfg.MODEL.BERT.HIDDEN_SIZE, 
                    (cfg.MODEL.BERT.HIDDEN_SIZE * cfg.MODEL.CLIPCAP.PROMPT_LEN) // 2,
                    cfg.MODEL.BERT.HIDDEN_SIZE * cfg.MODEL.CLIPCAP.PROMPT_LEN
                )
            )
            print("Using MLP as Mapper")

        else:
            # self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
            #                                                          clip_length, num_layers)
            # print("Using Transformer as Mapper")
            raise NotImplementedError

        return {
            "pre_encoder": pre_encoder,
            "mapper": mapper,
            "prompt_len": cfg.MODEL.CLIPCAP.PROMPT_LEN
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.CLIPCAP = CN()
        cfg.MODEL.CLIPCAP.PRE_ENCODER = 'GTransformerEncoder'
        cfg.MODEL.CLIPCAP.MAPPER_TYPE = 'MLP'
        cfg.MODEL.CLIPCAP.PROMPT_LEN = 10

    def forward(self, batched_inputs, mode=None):
        ret = self.pre_encoder(batched_inputs, mode)

        if mode == None or mode == 'v':
            gfeat = ret[kfg.ATT_FEATS][:, 0]
            batch_size = gfeat.shape[0]
            v_prompt = self.mapper(gfeat).view(batch_size, self.prompt_len, -1)
            ret.update({ kfg.ATT_FEATS: v_prompt })

            #### NOTE: need to re-build attn-mask here
            vmasks = v_prompt.new_ones(batch_size, self.prompt_len)
            vmasks = vmasks.to(dtype=next(self.parameters()).dtype)
            vmasks = vmasks.unsqueeze(1).unsqueeze(2)
            ext_vmasks = (1.0 - vmasks) * -10000.0
            ret.update({
                kfg.ATT_MASKS: vmasks,
                kfg.EXT_ATT_MASKS: ext_vmasks
            })

        return ret
    
    