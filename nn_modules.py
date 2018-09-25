import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


#   the general decoder RNN model
class AttentionDecoder(nn.Module):
    def __init__(self,output_size,embed_size,hidden_size,dropout=0.2,bidirectional=False,batch_first=True,MAX_LEN=150):
        super(AttentionDecoder,self).__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.embed_size=embed_size
        self.dropout=dropout
        self.max_len=MAX_LEN
        self.bidirectional=bidirectional
        self.batch_first=batch_first

        #   the embedding layer, transfer the one-hot word vector into a embedding representation (dense)
        self.embed_layer = nn.Embedding(output_size, embed_size)

        #   the attention machanism (input hidden information, output: max_length, kind of probabilistic output)
        self.att_on_context=nn.Linear(self.hidden_size * 2, self.max_len)

        #   after applied the attention, we need another no-linear transform as the input to the gru
        self.att_after_concat=nn.Linear(self.hidden_size * 2, self.hidden_size)

        #   the real GRU
        self.gru=nn.GRU(input_size=self.embed_size,hidden_size=self.hidden_size,num_layers=1,
                        dropout=self.dropout,bidirectional=self.bidirectional,batch_first=self.batch_first)

        #   the transformation to output
        self.out_layer=nn.Linear(self.hidden_size,self.output_size)

        return

    def forward(self,x,hidden,src_encodings):

        srcs=src_encodings.permute(1,0,3,2)
        srcs=srcs.squeeze(3).cuda()

        #   first calculate the embeddings (N*256) vector
        x_embed=self.embed_layer(x)

        #   concatenate the previous hidden state and the current word's embedding,
        #   outputs the attention weight (embedding size: hidden_size hidden: hidden_size)
        #   weight rely on input word and last hidden.
        hidden=hidden[0]

        x_hid_concat=torch.cat((x_embed,hidden),1)    # N*(256+256)

        #   apply the attention
        att_w=F.softmax(self.att_on_context(x_hid_concat),dim=1).unsqueeze(1)     # N*1*150 (64*1*150)
        src_after_att=torch.bmm(att_w,srcs)                                       # src_encodings: T*N*1*256, srcs: 64*150*256, out: 64*1*256 (the final encoded)

        #   concate the output to the original embedding
        x_and_src_after_att=torch.cat((x_embed.unsqueeze_(1),src_after_att),2)  # 64*1*512 (batch first mode)
        x_and_src_after_att=self.att_after_concat(x_and_src_after_att)
        x_and_src_after_att=F.relu(x_and_src_after_att)

        #   finally we can input to GRU
        if not self.batch_first:
            x_and_src_after_att=x_and_src_after_att.permute(1,0,2)

        x_out,x_hidden=self.gru(x_and_src_after_att,hidden.unsqueeze_(0))  # x_hidden: 1*64*256 (seq_len) first, x_out: 64*1*256

        #   we need to apply a softmax on it
        x_prob=F.log_softmax(self.out_layer(x_out.squeeze(1)),dim=1)
        return x_prob,x_hidden





#   the general encoder RNN model
class Encoder(nn.Module):
    def __init__(self,input_size,embed_size,hidden_size,dropout=0.2,bidirectional=False,batch_first=True):
        '''

        :param input_size:  the input size, i.e, number of input language words
        :param embed_size:  the embedding size, transfer the sparse input into a dense embedding
        :param hidden_size: the hidden features for GRU/LSTM
        '''
        super(Encoder,self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.embed_size=embed_size
        self.dropout=dropout
        self.bidirectional=bidirectional
        self.batch_first=batch_first

        #   the embedding layer, transfer the one-hot word vector into a embedding representation (dense)
        self.embed_layer=nn.Embedding(input_size,embed_size)

        #   the GRU, real thing
        self.gru=nn.GRU(input_size=embed_size, hidden_size=hidden_size,
                        num_layers=1,dropout=self.dropout,
                        bidirectional=self.bidirectional,batch_first=self.batch_first)

    # the forward stage
    def forward(self,x,hidden=None):

        #   convert x to embedding x (N*256) vector (batch first)
        x_embed=self.embed_layer(x)

        if self.batch_first:
            x_embed=x_embed.unsqueeze_(1)  # N*1*256
        else:
            x_embed=x_embed.unsqueeze_(0) # 1*N*256

        #   current output and the hidden state
        x_out,x_hidden=self.gru(x_embed,hidden)  # x_out: N*1*256, x_hidden: 1*N*256
        return x_out,x_hidden    # x_out: N*1*256, x_hidden: 1*N*256


