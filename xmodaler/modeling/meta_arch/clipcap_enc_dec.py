# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import pad_tensor, dict_to_cuda
from ..predictor import build_v_predictor
from .base_enc_dec import BaseEncoderDecoder
from .build import META_ARCH_REGISTRY

__all__ = ["ClipCapEncoderDecoder"]

@META_ARCH_REGISTRY.register()
class ClipCapEncoderDecoder(BaseEncoderDecoder):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        max_seq_len,
        token_embed,
        visual_embed,
        encoder,
        decoder,
        predictor,
        greedy_decoder,
        beam_searcher,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            token_embed=token_embed,
            visual_embed=visual_embed,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            greedy_decoder=greedy_decoder,
            beam_searcher=beam_searcher
        )

    def get_extended_attention_mask(self, batched_inputs):
        if kfg.TOKENS_MASKS not in batched_inputs:
            batched_inputs[kfg.TOKENS_MASKS] = torch.ones((batched_inputs[kfg.ATT_MASKS].size(0), self.max_seq_len)).cuda()

        tmasks = batched_inputs[kfg.TOKENS_MASKS]
        seq_length = tmasks.size(-1)
        tmasks = tmasks.to(dtype=next(self.parameters()).dtype)
        ext_u_tmasks = tmasks.unsqueeze(1).unsqueeze(2)
        ext_u_tmasks = (1.0 - ext_u_tmasks) * -10000.0

        ext_g_tmasks = torch.tril(torch.ones(
            (seq_length, seq_length), dtype=tmasks.dtype, device=tmasks.device))
        ext_g_tmasks = ext_g_tmasks.unsqueeze(0).expand(
            (tmasks.size(0), seq_length, seq_length))
        ext_g_tmasks = ext_g_tmasks * tmasks.unsqueeze(1)
        ext_g_tmasks = ext_g_tmasks.to(dtype=next(self.parameters()).dtype)
        ext_g_tmasks = ext_g_tmasks.unsqueeze(1)
        ext_g_tmasks = (1.0 - ext_g_tmasks) * -10000.0

        vmasks = batched_inputs[kfg.ATT_MASKS]
        vmasks = vmasks.to(dtype=next(self.parameters()).dtype)
        vmasks = vmasks.unsqueeze(1).unsqueeze(2)
        ext_vmasks = (1.0 - vmasks) * -10000.0

        return {
            kfg.TOKENS_MASKS: tmasks,
            kfg.EXT_U_TOKENS_MASKS: ext_u_tmasks,
            kfg.EXT_G_TOKENS_MASKS: ext_g_tmasks,
            kfg.ATT_MASKS: vmasks,
            kfg.EXT_ATT_MASKS: ext_vmasks
        }

    def _forward(self, batched_inputs):
        inputs = batched_inputs
        masks = self.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = self.visual_embed(batched_inputs)
        inputs.update(ve_out)

        if self.encoder is not None:
            encoder_out_v = self.encoder(inputs, mode='v')
            inputs.update(encoder_out_v)

        if self.decoder is not None:
            inputs = self.decoder.preprocess(inputs)

        te_out = self.token_embed(batched_inputs)
        inputs.update(te_out)
        
        if self.encoder is not None:
            encoder_out_t = self.encoder(inputs, mode='t')
            inputs.update(encoder_out_t)
        
        if self.decoder is not None:
            decoder_out = self.decoder(inputs)
            inputs.update(decoder_out)

        if self.predictor is not None:
            tlogits = self.predictor(inputs)
            inputs.update(tlogits)

        return inputs
    

# class ClipCaptionModel(nn.Module):
    
#     def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
#                  num_layers: int = 8, mapping_type: str = "mlp"):
#         super(ClipCaptionModel, self).__init__()
#         self.prefix_length = prefix_length
#         self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#         self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
#         if mapping_type == "mlp":
#             self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
#                                      self.gpt_embedding_size * prefix_length))
#             print("Using MLP as Mapper")
#         else:
#             self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
#                                                                      clip_length, num_layers)
#             print("Using Transformer as Mapper")


#     def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
#         return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

#     def forward(self, *args, **kwargs):
#         is_decode = kwargs.get('is_decode', False)
#         if is_decode:
#             return self.generate_cap(*args, **kwargs)
#         else:
#             return self.encode_forward(*args, **kwargs)    

#     def encode_forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
#                 labels: Optional[torch.Tensor] = None):
#         embedding_text = self.gpt.transformer.wte(tokens)
#         prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
#         embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
#         if labels is not None:
#             dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
#             labels = torch.cat((dummy_token, tokens), dim=1)
#         out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
#         return out            

#     def generate_cap(self, img_feats, max_length=None,
#             do_sample=None, num_beams=None, temperature=None, top_k=None, top_p=None,
#             repetition_penalty=None, eos_token=None, length_penalty=None,
#             num_return_sequences=None, num_keep_best=1, is_decode=None
#             ):
#         """ Generates captions given image features
#         """
#         # TODO: self.num_keep_best is given in generate, then search func should not use magic number 
#         assert is_decode
#         batch_size = img_feats.shape[0]
#         self.num_keep_best = num_keep_best

#         ## project prefix to GPT space
#         # [bs, pre_len, emb_size]
#         img_feats = self.clip_project(img_feats).view(-1, self.prefix_length, self.gpt_embedding_size)

#         cur_len = 0
#         if  num_return_sequences != 1:
#             # Expand input to num return sequences
#             img_feats = self._expand_for_beams(img_feats, num_return_sequences)
#             effective_batch_size = batch_size * num_return_sequences
#         else:
#             effective_batch_size = batch_size

#         output = self.generate(
#             img_feats,
#             cur_len,
#             max_length,
#             do_sample,
#             temperature,
#             top_k,
#             top_p,
#             repetition_penalty,
#             eos_token,
#             effective_batch_size,
#         )

#         return output

#     def generate(
#             self,
#             input_embeds,
#             cur_len,
#             max_length,
#             do_sample,
#             temperature,
#             top_k,
#             top_p,
#             repetition_penalty,
#             eos_token,
#             batch_size,
#         ):
#             """ Generate sequences for each example without beam search (num_beams == 1).
#                 All returned sequence are generated independantly.
#             """
#             self.num_keep_best = 1
#             assert self.num_keep_best == 1, 'cannot generate >1 sentences in greedy search'
#             # current position / max lengths / length of generated sentences / unfinished sentences
#             unfinished_sents = []
#             #[bs] = 1
#             cur_unfinished = input_embeds.new(batch_size).fill_(1)
#             eos_token_id = self.tokenizer.encode(eos_token)[0]

#             # log of scores for each sentence in the batch
#             logprobs = []

#             past = None

#             token_ids = None

#             pad_token_id = self.tokenizer.encode('<|endoftext|>')[0]

#             while cur_len < max_length:
#                 if past:
#                     outputs = self.gpt(input_ids=next_token.unsqueeze(-1), past_key_values=past)
#                 else:
#                     outputs = self.gpt(inputs_embeds=input_embeds, past_key_values=past)
#                 # [bs, pre_len, 768] -> [bs, voc_size]
#                 next_token_logits = outputs.logits[:, -1, :]

#                 # if model has past, then set the past variable to speed up decoding
#                 if self._do_output_past(outputs):
#                     past = outputs.past_key_values

#                 # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
#                 if repetition_penalty != 1.0 and token_ids is not None:
#                     for i in range(batch_size):
#                         for previous_token in set(token_ids[i].tolist()):
#                             # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
#                             if next_token_logits[i, previous_token] < 0:
#                                 next_token_logits[i, previous_token] *= repetition_penalty
#                             else:
#                                 next_token_logits[i, previous_token] /= repetition_penalty

#                 if do_sample:
#                     # Temperature (higher temperature => more likely to sample low probability tokens)
#                     if temperature != 1.0:
#                         next_token_logits = next_token_logits / temperature
#                     # Top-p/top-k filtering
#                     next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
#                     # Sample
#                     # [bs * sample_time]
#                     next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
#                 else:
#                     # Greedy decoding
#                     # next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
#                     next_token = torch.argmax(next_token_logits, dim=-1)

#                 # Compute scores
#                 _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
#                 _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
#                 logprobs.append(_scores)  # (batch_size, 1)
#                 unfinished_sents.append(cur_unfinished)

#                 tokens_to_add = next_token * cur_unfinished + pad_token_id * (1 - cur_unfinished)
#                 if token_ids is None:
#                     token_ids = tokens_to_add.unsqueeze(-1)
#                 else:
#                     token_ids = torch.cat([token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                

#                 #for t in input_ids:
#                     #print(self.tokenizer.convert_ids_to_tokens(t.tolist()))
#                 cur_unfinished = cur_unfinished.mul(next_token.ne(eos_token_id).long())
#                 cur_len = cur_len + 1

#                 # stop when there is a </s> in each sentence, or if we exceed the maximul length
#                 if cur_unfinished.max() == 0:
#                     break

#             # add eos_token_ids to unfinished sentences
#             # NOTE: for OSCAR pretrained model, it should be ended with SEP token. However we end with '.' to keep consistent with OSCAR
#             if cur_len == max_length:
#                 token_ids[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_id)

#             logprobs = torch.cat(logprobs, dim=1)
#             unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
#             sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
#             # return logprobs to keep consistent with beam search output
#             logprobs = sum_logprobs / unfinished_sents.sum(dim=1)
#             # (batch_size, n_best, max_len), (batch_size, n_best)

#             pad_len = max_length - token_ids.shape[1]
#             if pad_len > 0:
#                 padding_ids = token_ids.new(batch_size, pad_len).fill_(pad_token_id)
#                 token_ids = torch.cat([token_ids, padding_ids], dim=1)

#             # (batch_size, n_best, max_len), (batch_size, n_best)
#             return token_ids.unsqueeze(1), logprobs.unsqueeze(1)

#     def _expand_for_beams(self, x, num_expand):
#         if x is None or num_expand == 1:
#             return x
#         # x: [bs, len, embed_size]
#         input_shape = list(x.shape)
#         expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
#         # expanded_x: batch * num_expand * len * embed_size
#         x = x.unsqueeze(1).expand(expanded_shape)
#         # (batch_size * num_expand, ...)
#         x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
#         # x: (batch * num_expand) * len * embed_size
#         return x

#     def _do_output_past(self, outputs):
#         return len(outputs) > 1