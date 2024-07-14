# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os

def decode_sequence(vocab, seq, vocab_sem):
    N, T = seq.size()
    sents = []
    sents_sem = []
    for n in range(N):
        words = []
        words_sem = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                words.append('.')
                break

            word = vocab[ix]
            words.append(word)
            if vocab_sem is not None and word in vocab_sem:
                words_sem.append(word)

        sent = ' '.join(words)
        sent_sem = ', '.join(set(words_sem)) 

        sents.append(sent)
        sents_sem.append(sent_sem)
    return sents, sents_sem

def decode_sequence_bert(tokenizer, seq, sep_token_id):
    N, T = seq.size()
    seq = seq.data.cpu().numpy()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == sep_token_id:
                break
            words.append(tokenizer.ids_to_tokens[ix])
        sent = tokenizer.convert_tokens_to_string(words)
        sents.append(sent)
    return sents