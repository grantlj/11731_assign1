import math
from typing import List

import numpy as np
import copy
import torch

def save_model_by_state_dict(model,model_fn):
    pass
    torch.save({'state_dict':model.cpu().state_dict()},model_fn)
    model=model.cuda()
    return

#   load model state dict
def load_model_by_state_dict(model,state_dict_fn):
    pass
    model_dict=torch.load(state_dict_fn)
    model.load_state_dict(model_dict['state_dict'])
    return model


def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into 
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t

def input_transpose_max_len(sents, pad_token,MAX_LEN):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch

    We abandon all those sentenses with length>max_len
    """

    batch_size = len(sents)

    sents_t = []
    for i in range(MAX_LEN):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

#  LJ: A method to yield the training batches (NOTE: the shuffle is False here)
def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

'''
def convert_label_list_to_one_hot_np_mat(label_list,num_labels):
    batch_size=len(label_list)
    ret_mat=np.zeros((batch_size,num_labels))

    for i in range(0,len(label_list)):
        ret_mat[i][label_list]=1
    return ret_mat
'''