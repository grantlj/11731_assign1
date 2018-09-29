'''
    09/28/2018: Generating the initial word embeddings from the pre-train models.
'''

import os
import sys
sys.path.append("../")
import pickle
import numpy as np
from vocab import Vocab, VocabEntry

#   english embedding
en_word_embed_fn="word_embedding/wiki.en.vec"
en_word_mat_fn="word_embedding/en.npz"

#   germany embedding
de_word_embed_fn="word_embedding/wiki.de.vec"
de_word_mat_fn="word_embedding/de.npz"

#   vocab fn
vocab_fn="data/vocab.bin"

assert os.path.isfile(en_word_embed_fn)
assert os.path.isfile(de_word_embed_fn)
assert os.path.isfile(vocab_fn)

DIM=300

def get_embedding_mat(embed_fn,vocab):
    size=len(vocab)
    dim=DIM
    ret_mat=np.random.rand(size,dim)

    #ret_mat = np.zeros((size, dim))
    not_found_size=size

    not_exist=0
    with open(embed_fn,"r") as f:
        line_ind=0
        while True:
            line_ind+=1
            cur_line=f.readline()
            if cur_line=="" or cur_line is None:
                break
            if line_ind==1:
                continue

            cur_line=cur_line.replace("\n","")
            cur_line=cur_line.replace("\r","")
            tmp_str=cur_line.split(" ")

            word=tmp_str[0];embed=tmp_str[1:len(tmp_str)-1]
            embed=[float(x) for x in embed]
            embed=np.asarray(embed)

            if not word in vocab.word2id:
                #not_exist+=1
                #print("Word: ",word," not existed...")
                continue

            word_indice=vocab.word2id[word]
            ret_mat[word_indice,:]=embed
            not_found_size-=1
            assert len(embed)==dim

    print("not existed ratio:",float(not_found_size),"/",float(size))
    return ret_mat


if __name__=="__main__":
    all_vocab=pickle.load(open(vocab_fn, 'rb'))

    de_vocab=all_vocab.src
    en_vocab=all_vocab.tgt

    de_mat=get_embedding_mat(de_word_embed_fn,de_vocab)
    np.savez_compressed(de_word_mat_fn,embedding=de_mat)

    en_mat=get_embedding_mat(en_word_embed_fn,en_vocab)
    np.savez_compressed(en_word_mat_fn, embedding=en_mat)

    print("done.")