# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode --vocab=<file> [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode --vocab=<file> [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
from collections import namedtuple
from typing import Any

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import utils
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry
import torch
import torch.nn.functional as F
import pdb
from torch import Tensor
from nn_modules import Encoder,AttentionDecoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils import clip_grad_norm


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

MAX_LEN=100

class NMT(object):

    def __init__(self, embed_size, hidden_size, vocab, loss, dropout_rate=0.2,decoding_type="ATTENTION",bi_direct=True):
        super(NMT, self).__init__()

        self.bi_direct = bi_direct

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.decoding_type=decoding_type
        self.loss=loss
        self.tgt_vocab_size = len(vocab.tgt.word2id)
        self.src_embed = nn.Embedding(len(vocab.src.word2id), embed_size, padding_idx=0).cuda()
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, embed_size, padding_idx=0).cuda()
        self.src_embed.weight.requires_grad = False
        self.out = nn.Linear(hidden_size, embed_size).cuda()
        self.word_dist = nn.Linear(embed_size, self.tgt_vocab_size).cuda()
        self.word_dist.weight = self.tgt_embed.weight

        self.key_size = 50



        #if self.decoding_type is "ATTENTION":
        '''
        self.encoder=Encoder(embed_size=embed_size,input_size=len(self.vocab.src),
                                 hidden_size=hidden_size,dropout=dropout_rate,batch_first=True).cuda()
        '''
        self.encoder = nn.LSTM(embed_size, hidden_size,bidirectional=self.bi_direct).cuda()

        if self.bi_direct:
            self.decoder = nn.LSTM(embed_size * 2, hidden_size*2).cuda()
            self.a_key = nn.Linear(hidden_size*2, self.key_size).cuda()
            self.q_key = nn.Linear(hidden_size*2, self.key_size).cuda()
            self.q_value = nn.Linear(hidden_size*2, embed_size).cuda()

        else:
            self.decoder = nn.LSTM(embed_size * 2, hidden_size).cuda()
            self.a_key = nn.Linear(hidden_size, self.key_size).cuda()
            self.q_key = nn.Linear(hidden_size, self.key_size).cuda()
            self.q_value = nn.Linear(hidden_size, embed_size).cuda()

        # initialize neural network layers...
        '''
        if decoding_type=="ATTENTION":
            self.decoder=AttentionDecoder(embed_size=embed_size,
                                          output_size=len(self.vocab.tgt),
                                          hidden_size=hidden_size,dropout=dropout_rate,batch_first=True,MAX_LEN=MAX_LEN).cuda()
        '''

    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]],keep_grad=True) -> Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """

        #   in the first place, let us padding the original inputs to fixed length (MAX_TOKEN_LEN*BATCH_SIZE)
        '''
        src_sents_padded=utils.input_transpose_max_len(src_sents,pad_token='<pad>',MAX_LEN=MAX_LEN)
        tgt_sents_padded=utils.input_transpose_max_len(tgt_sents,pad_token='<pad>',MAX_LEN=MAX_LEN)
        '''
        pairs = list(zip(src_sents, tgt_sents))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        src_sents, tgt_sents = zip(*pairs)
        src_lengths = [len(s) for s in src_sents]
        tgt_lengths = [len(s) for s in tgt_sents]
        src_max_len = max(src_lengths)
        tgt_max_len = max(tgt_lengths)
        batch_size = len(src_sents)
        src_ind = torch.zeros(batch_size, src_max_len).long()
        tgt_ind = torch.zeros(batch_size, tgt_max_len).long()
        for x in range(len(src_sents)):
            src_ind[x, :len(src_sents[x])] = torch.LongTensor(self.vocab.src.words2indices(src_sents[x]))
            tgt_ind[x, :len(tgt_sents[x])] = torch.LongTensor(self.vocab.tgt.words2indices(tgt_sents[x]))
        src_ind = src_ind.cuda()
        tgt_ind = tgt_ind.cuda()

        if keep_grad:
            #   for training stage
            #src_encodings, decoder_init_state = self.encode(src_sents_padded)
            src_encodings, decoder_init_state = self.encode(src_ind, src_lengths)
            loss, num_words = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, tgt_lengths)
        else:
            #   for test stage
            with torch.no_grad():
                #src_encodings, decoder_init_state = self.encode(src_sents_padded)
                src_encodings, decoder_init_state = self.encode(src_ind, src_lengths)
                loss, num_words = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, tgt_lengths)

        return loss, num_words

    def init_hidden(self, hidden):
        return hidden

    def encode(self, src_sents: List[List[str]], src_lengths, keep_grad=True) -> Tuple[Tensor, Any]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable 
                with shape (batch_size, source_sentence_length, encoding_dim), or in other formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """

        '''
        sent_len=len(src_sents);batch_size=len(src_sents[0])

        #   iterate through the step-th words in all sentences
        hidden=None

        src_encodings=torch.zeros([sent_len,batch_size,1,self.hidden_size]).cuda()  #  T*N*1*256

        #   each step at one time
        for step in range(0,sent_len):
            x=self.vocab.src.words2indices(src_sents[step])

            #   the batch_size*1 vector (the indices)
            x=torch.from_numpy(np.asarray(x)).type('torch.LongTensor').cuda()

            #  enc_out: N*1*256, x_hidden: 1*N*256
            enc_out,hidden=self.encoder.forward(x,hidden)
            src_encodings[step]=enc_out

        decoder_init_state=hidden                 #  the decoder's input hidden layer
        return src_encodings, decoder_init_state  # the coding for each state (sent_len,batch_size,1,256), the last hidden_output (1*batch_size*256)
        '''
        src_embed = self.src_embed(src_sents)
        packed_input = pack_padded_sequence(src_embed, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)


        if self.bi_direct:
            src_last_hidden[0]=torch.cat((src_last_hidden[0][0],src_last_hidden[0][1]),2)
            src_last_hidden[1]=torch.cat((src_last_hidden[1][0], src_last_hidden[1][1]), 2)

        decoder_hidden = self.init_hidden(src_last_hidden) 

        return src_hidden, decoder_hidden  #    src_hidden, in bi-drect mode : [48,31,512], decoder hidden: same hase the src hidden
        
    def attention(self, decoder_hidden, src_hidden, src_lengths, length):
        a_key = self.a_key(decoder_hidden[0].squeeze(0))

        q_key = self.q_key(src_hidden)
        q_value = self.q_value(src_hidden)
        q_energy = torch.bmm(q_key, a_key.unsqueeze(2)).squeeze(2)
        q_mask  = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        q_energy[~q_mask] = -np.inf
        q_weights = F.softmax(q_energy, dim=1).unsqueeze(1)
        #q_weights = F.sigmoid(q_energy).unsqueeze(1)
        q_context = torch.bmm(q_weights, q_value)

        return q_context


    def decode(self, src_encodings: Tensor, src_lengths, decoder_init_state: Any, tgt_sents: List[List[str]], tgt_lengths, keep_grad=True):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """

        '''
        cum_loss=0

        tgt_sent_len=len(tgt_sents);batch_size=len(tgt_sents[0])

        scores=[]
        hidden=decoder_init_state
        for step in range(0,tgt_sent_len-1):

            x = self.vocab.tgt.words2indices(tgt_sents[step])
            next_x=self.vocab.tgt.words2indices(tgt_sents[step+1])

            #   the batch_size*1 vector (the indices)
            x = torch.from_numpy(np.asarray(x)).type('torch.LongTensor').cuda()
            next_x = torch.from_numpy(np.asarray(next_x)).type('torch.LongTensor').cuda()

            #   batch_size*n_words
            dec_out,hidden=self.decoder.forward(x,hidden,src_encodings)
            scores.append(dec_out.cpu().data.numpy())

            #   cum loss at all steps
            cum_loss+=self.loss(dec_out,next_x)


        scores=np.asarray(scores)
        return scores,cum_loss
        '''
        batch_size = src_encodings.size(0)
        length = src_encodings.size(1)
        tgt_embed = self.tgt_embed(tgt_sents)
        tgt_l = tgt_embed.size(1)

        decoder_input = tgt_embed[:, 0, :].unsqueeze(1)
        decoder_outputs = torch.cuda.FloatTensor(batch_size, tgt_l - 1, self.tgt_vocab_size)
        decoder_hidden = decoder_init_state
        for step in range(tgt_l - 1):
            context = self.attention(decoder_hidden, src_encodings, src_lengths, length)
            decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1), decoder_hidden)
            decoder_outputs[:, step, :] = self.word_dist(self.out(decoder_output.squeeze(1)))
            decoder_input = tgt_embed[:, step+1, :].unsqueeze(1)

        logits = F.log_softmax(decoder_outputs, dim=2)
        logits = logits.contiguous().view(-1, self.tgt_vocab_size)
        loss = self.loss(logits, tgt_sents[:, 1:].contiguous().view(-1))
        return loss, (tgt_sents[:, 1:] != 0).sum().item()


    def beam_search(self, src_sent: List[str], beam_size: int=20, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        self.decoder.eval()
        self.encoder.eval()

        '''
        hypotheses=[]
        src_sents=[src_sent]
        src_sents_padded = utils.input_transpose_max_len(src_sents, pad_token='<pad>', MAX_LEN=MAX_LEN)

        #   run the encoding part for the encoding on the src sentences and the decoder's initial state

        #   src_encodings: 50*1*1*256, (step*1*1*hidden), decoder_init_state: 1*1*256
        src_encodings, decoder_init_state = self.encode(src_sents_padded)

        #   prev_beams: the whole sentences currently (the initial status should be the index of the <s>)
        prev_beams=[(0.0,[self.vocab.tgt.word2id['<s>']],decoder_init_state,[0.0])]

        #   a heap to store all the results
        max_beam_heap=[]
        for cur_step in range(max_decoding_time_step):

            new_beams=[]

            for prev_beam in prev_beams:
                #   handle a particular beam
                #   input the previous status to the decoder

                prev_score=prev_beam[0]
                prev_sent_words=prev_beam[1][:];prev_hidden=prev_beam[2];prev_scores=prev_beam[3][:]
                prev_last_word=[prev_sent_words[-1]]

                #   the 1*1 vector (the indices)
                prev_last_word = torch.from_numpy(np.asarray(prev_last_word)).type('torch.LongTensor').cuda()
                dec_out, cur_hidden = self.decoder.forward(prev_last_word, prev_hidden, src_encodings)

                #   find top-k in dec_out
                top_k_val,top_k_ind=torch.topk(dec_out,k=beam_size)

                top_k_val=np.squeeze(top_k_val.detach().cpu().numpy())
                top_k_ind=np.squeeze(top_k_ind.detach().cpu().numpy())

                for prob_val,prob_ind in zip(top_k_val,top_k_ind):

                    cur_sents_words=prev_sent_words[:]
                    cur_sents_words.append(prob_ind)
                    cur_scores=prev_scores[:]
                    cur_scores.append(prob_val)
                    cur_score=prev_score+prob_val

                    if prob_ind==self.vocab.tgt.word2id['</s>']:
                        #   the sentence is ended, normalize the score by length of the sentences
                        new_beam = (-cur_score/len(cur_sents_words), cur_sents_words, cur_hidden, cur_scores)
                        heapq.heappush(max_beam_heap,new_beam)

                    else:
                        #   we should continue grow this beam
                        new_beam = (cur_score, cur_sents_words, cur_hidden, cur_scores)
                        new_beams.append(new_beam)

            if len(new_beams)==0:
                #   there are no new beams rest, break the searching process
                break

            #   after handling all previous beams, we output the probs
            new_beams=sorted(new_beams,reverse=True)[0:beam_size]
            prev_beams=new_beams[:]

        #   push the prev_beams into the heap
        for cur_beam in prev_beams:
            cur_score=cur_beam[0];cur_sents_words=cur_beam[1];cur_hidden=cur_beam[2];cur_scores=cur_beam[3]
            cur_score=-cur_score/len(cur_sents_words)
            new_beam=(cur_score,cur_sents_words,cur_hidden,cur_scores)
            heapq.heappush(max_beam_heap,new_beam)


        #   finally, we convert the beams into hypotheses
        for i in range(0,beam_size):
            cur_beam=heapq.heappop(max_beam_heap)
            cur_score = -cur_beam[0];cur_sents_ind = cur_beam[1]

            #   convert words to sentences string (add a ending string if necessary)
            if not cur_sents_ind[-1]==self.vocab.tgt.word2id['</s>']:
                cur_sents_ind.append(self.vocab.tgt.word2id['</s>'])

            cur_sents_words=self.vocab.tgt.indices2words(cur_sents_ind)
            cur_hyp=Hypothesis(value=cur_sents_words,score=cur_score)
            hypotheses.append(cur_hyp)


        return hypotheses
        '''

        self.eou = 2
        top_k = 15
        src_ind = torch.cuda.LongTensor(self.vocab.src.words2indices(src_sent))
        src_embed = self.src_embed(src_ind).unsqueeze(0)
        src_lengths = np.asarray([len(src_sent)])
        packed_input = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden) 

        batch_size = src_embed.size(0)
        assert batch_size == 1 
        eos_filler = torch.zeros(beam_size).long().cuda().fill_(self.eou)
        decoder_input = self.tgt_embed(torch.cuda.LongTensor([1])).unsqueeze(1)
        length = src_hidden.size(1)

        context = self.attention(decoder_hidden, src_hidden, src_lengths, length)
        decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
        decoder_output = self.word_dist(self.out(decoder_output.squeeze(1)))
        decoder_output[:, 0] = -np.inf

        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        beam = torch.zeros(beam_size, max_decoding_time_step).long().cuda()
        beam[:, 0] = argtop
        beam_probs = logprobs.clone().squeeze(0)
        beam_eos = argtop.squeeze(0) == self.eou
        decoder_hidden = (decoder_hidden[0].expand(1, beam_size, self.hidden_size).contiguous(),
                          decoder_hidden[1].expand(1, beam_size, self.hidden_size).contiguous())
        decoder_input = self.tgt_embed(argtop.squeeze(0)).unsqueeze(1)
        src_hidden = src_hidden.expand(beam_size, length, self.hidden_size)

        for t in range(max_decoding_time_step):
            context = self.attention(decoder_hidden, src_hidden, src_lengths, length)
            decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1), decoder_hidden)
            decoder_output = self.word_dist(self.out(decoder_output))

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(0), dim=1), top_k, dim=1)
            best_probs, best_args = (beam_probs.expand(top_k, beam_size).transpose(0, 1) + logprobs).view(-1).topk(beam_size)

            last = best_args / top_k
            curr = best_args % top_k
            beam[:, :] = beam[last, :]
            beam_eos = beam_eos[last]
            beam_probs = beam_probs[last]
            pdb.set_trace()
            beam[:, t+1] = argtop.view(_batch_size, beam_size, top_k)[x][last.data, curr.data] * Variable(~beam_eos[x]).long() + eos_filler * Variable(beam_eos[x]).long()
            mask = torch.cuda.ByteTensor(_batch_size, beam_size).fill_(0)
            mask[x] = ~beam_eos[x]
            beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
            decoder_hidden[0][:, x, :, :] = decoder_hidden[0][:, x, :, :][:, last, :]
            decoder_hidden[1][:, x, :, :] = decoder_hidden[1][:, x, :, :][:, last, :]

            beam_eos = beam_eos | (beam[:, :, t+1] == self.eou).data
            decoder_hidden = (decoder_hidden[0].view(1, _batch_size * beam_size, -1),
                              decoder_hidden[1].view(1, _batch_size * beam_size, -1))
            decoder_input = self.embed(beam[:, :, t+1].contiguous().view(-1)).unsqueeze(1)

            if beam_eos.all():
                break

        best, best_arg = beam_probs.max(1)
        generations = beam[torch.arange(_batch_size).long().cuda(), best_arg.data].data.cpu()
        return generations


    def evaluate_ppl(self, dev_data: List[Any], batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """

        self.set_model_to_eval()

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss, num_words = self.__call__(src_sents, tgt_sents,keep_grad=False)

            loss=loss.detach().cpu().numpy()

            cum_loss += loss * num_words
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return cum_loss,ppl

    #   set model to train and test state
    def set_model_to_train(self):
        self.encoder.train()
        self.decoder.train()
        return

    #   set model to validation
    def set_model_to_eval(self):
        self.encoder.eval()
        self.decoder.eval()
        return

    def load(self,model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        encoder_fn = model_path.replace(".bin", "_encoder.pkl")
        decoder_fn = model_path.replace(".bin", "_decoder.pkl")

        self.encoder=utils.load_model_by_state_dict(self.encoder,encoder_fn)
        self.decoder=utils.load_model_by_state_dict(self.decoder,decoder_fn)

        self.encoder.cuda().eval()
        self.decoder.cuda().eval()

        return

    def save(self, path: str):
        """
        Save current model to file
        """
        encoder_fn=path.replace(".bin","_encoder.pkl")
        decoder_fn=path.replace(".bin","_decoder.pkl")

        utils.save_model_by_state_dict(self.encoder,encoder_fn)
        utils.save_model_by_state_dict(self.decoder,decoder_fn)

        return



def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


#   LJ: the training process starting here.
def train(args: Dict[str, str]):

    #   LJ: source corpus and target corpus
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    #   LJ: the validation set (source and target)
    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt =    read_corpus(args['--dev-tgt'], source='tgt')

    #   LJ: the training and validation sentences pairs
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    #   LJ: the configurations
    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    #   LJ: read the vocabulary
    vocab = pickle.load(open(args['--vocab'], 'rb'))

    #   LJ: set up the loss function (ignore to <pad>)
    nll_loss = nn.NLLLoss(ignore_index=0)

    src_vectors = np.load('word_embedding/de.npz')['embedding']
    tgt_vectors = np.load('word_embedding/en.npz')['embedding']

    #   LJ: build the model
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab,loss=nll_loss)
    model.src_embed.weight.data = torch.from_numpy(src_vectors).float().cuda()
    model.tgt_embed.weight.data = torch.from_numpy(tgt_vectors).float().cuda()

    #   LJ: the learning rate
    lr = float(args['--lr'])


    #   LJ: setting some initial losses, etc.
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    #   LJ: setup the optimizer
    optimizer = optim.Adam(list(model.encoder.parameters())+list(model.decoder.parameters()), lr=lr)

    while True:

        #   start the epoch
        epoch += 1

        #   LJ: ok, we yield the sentences in a shuffle manner.
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):

            model.set_model_to_train()

            train_iter += 1

            #   LJ: current batch size
            batch_size = len(src_sents)

            # (batch_size)
            # LJ: train on the mini-batch and get the loss, backpropagation

            #loss = -model(src_sents, tgt_sents)
            optimizer.zero_grad()
            loss, num_words = model(src_sents, tgt_sents)
            loss.backward()
            clip_grad_norm(list(model.encoder.parameters())+list(model.decoder.parameters()),clip_grad)
            optimizer.step()


            #   add the loss to cumlinative loss
            report_loss += loss.detach().cpu().numpy() * num_words
            cum_loss += loss.detach().cpu().numpy() * num_words

            #   LJ: how many targets words are there in all target sentences in current batch
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`

            #   LJ: all cumulative words
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict

            #   LJ: all number of instances handled
            report_examples += batch_size
            cumulative_examples += batch_size

            #   LJ: print out the training loss
            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                model.set_model_to_eval()
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                #   LJ: the validation is implemented in a seperate function
                cum_loss,dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                #valid_metric = -dev_ppl
                valid_metric=-cum_loss


                print('validation: iter %d, dev. ppl %f, val cum loss: %f' % (train_iter, dev_ppl,cum_loss), file=sys.stderr)

                #   LJ: a new better model is found.
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # You may also save the optimizer's state, adjust the training weight, since we found there are too
                    #   much iterations without improvements.
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model.load(model_save_path)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:


    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


#   LJ: the decode step (i.e., the test stage)
def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)

    #   LJ: read the vocabulary
    vocab = pickle.load(open(args['--vocab'], 'rb'))

    #   LJ: set up the loss function (ignore to <pad>)
    nll_loss = nn.NLLLoss(ignore_index=0)

    #   LJ: build the model
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab, loss=nll_loss)

    model.load(args["MODEL_PATH"])

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)
    print(args)
    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
