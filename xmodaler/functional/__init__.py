# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .func_io import (
    read_lines,
    read_lines_set,
    load_vocab,
    read_np,
    read_np_bbox
)

from .func_feats import (
    iou,
    boxes_to_locfeats,
    dict_as_tensor,
    dict_to_cuda,
    pad_tensor,
    expand_tensor,
    clip_v_inputs,
    clip_t_inputs
)

from .func_caption import (
    decode_sequence,
    decode_sequence_bert
)

from .func_pretrain import (
    random_word,
    random_region,
    caption_to_mask_tokens
)

from .func_others import (
    flat_list_of_lists,
    noise_injection
)

from .func_model import (
    build_clip_model,
    build_clip_from_xmodaler,
    get_img_txt_similarity
)