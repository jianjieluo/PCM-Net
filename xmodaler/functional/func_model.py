import torch
import torch.nn as nn
import torch.nn.functional as F

import clip


def build_clip_model(clip_model_name_or_path, context_length):
    # load clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load(clip_model_name_or_path, jit=False, device=device)
    # resize context_length in clip_model
    clip_model.context_length = context_length
    clip_model.positional_embedding = nn.parameter.Parameter(clip_model.positional_embedding.data[:context_length])
    new_attn_mask = clip_model.build_attention_mask().to(device)
    for layer in clip_model.transformer.resblocks:
        layer.attn_mask = new_attn_mask
    return clip_model, clip_preprocess


def build_clip_from_xmodaler(clip_model_name_or_path, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    weights = ckpt['trainer']['ema']
    weights = {key[len('module.module.clip_model.'):] if key.startswith('module.module.clip_model.') else key : value for key, value in weights.items()}
    context_length = weights['positional_embedding'].data.size(0)

    clip_model, clip_preprocess = build_clip_model(clip_model_name_or_path, context_length)
    for (k1,v1), (k2,v2) in zip(clip_model.named_parameters(), weights.items()):
        assert (k1 == k2)
        assert (v1.shape == v2.shape)

    clip_model.load_state_dict(weights)
    return clip_model, clip_preprocess


def get_img_txt_similarity(image_features, sents, cand_size, clip_model, device, use_softmax=True):
    # tokenize by clip_tokenizer
    text = clip.tokenize(sents, context_length=clip_model.context_length, truncate=True).to(device)
    # calc the sent embed
    with torch.no_grad():
        text_features = clip_model.encode_text(text).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(-1, cand_size, text_features.size(-1))
        # calc the clip similarity with img feature
        similarity = torch.einsum('bid,btd->bit', image_features, text_features).squeeze(1)

    if use_softmax:
        logit_scale = clip_model.logit_scale.exp()
        similarity = logit_scale * similarity
        similarity = F.softmax(similarity, dim=-1)

    return similarity