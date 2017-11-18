import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import string
import os
import json

class Character_LSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0):
        super(Character_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        
    def forward(self, sentence, length):
        x = self.embedding(sentence)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)
        lstm_out, (hn, cn) = self.lstm(x)
        return hn.view(hn.size(1), -1)


class Character_CNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes):
        super(Character_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k, padding=k-1) for k in filter_sizes])

    def forward(self, x):
        # num_sent, sent_len, word_len
        num_sent, sent_len, word_len = x.size()
        x = self.embedding(x.view(-1, word_len))
        x = x.permute(0, 2, 1)
        x = [F.tanh(conv(x)) for conv in self.convs]  # [**, num_filters, **]
        x = [F.max_pool1d(sub_x, sub_x.size(2)).squeeze(2) for sub_x in x]  # [**, num_filters]
        x = torch.cat(x, 1).view(num_sent, sent_len, -1)
        return x


class Word_LSTM(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Word_LSTM, self).__init__()
        self.lstm = nn.RNN(input_dim, output_dim/2, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True, bidirectional=False)

    def forward(self, x):
        # output, (hn, cn) = self.lstm(x)
        output, hn = self.lstm(x)
        return output


class GNN_Layer(nn.Module):

    def __init__(self, input_dim, output_dim, model, globalnode):
        super(GNN_Layer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear_ud = nn.Linear(input_dim, output_dim)
        if model == 'gnn':
            self.linear_lr = nn.Linear(input_dim, output_dim)
        self.model = model
        self.globalnode = globalnode
        if self.globalnode:
            self.linear_g  = nn.Linear(input_dim, output_dim)
            self.linear_go = nn.Linear(output_dim, output_dim)

    def sparse_mm(self, a, x):
        res = Variable(torch.zeros(x.size())).cuda()
        for i in range(a.data._values().size(0)):
            u = a.data._indices()[0, i]
            v = a.data._indices()[1, i]
            res[u] = res[u] + a.data._values()[i] * x[v]
        return res

    def forward(self, x, mask, a_ud, a_lr):
        # x: (N, input_dim)
        num_sent, sent_len, hidden = x.size()
        x = x.contiguous().view(-1, hidden)
        if self.model == 'gnn':
            h_ud = torch.mm(a_ud, x)
            h_lr = torch.mm(a_lr, x)
            h = self.linear(x) + self.linear_ud(h_ud) + self.linear_lr(h_lr) 
            if self.globalnode:
                n = mask.sum()
                h_g = mask.view(1,-1).matmul(F.relu(self.linear_g(x))) / n
                h = h + self.linear_go(h_g).expand(x.size(0), h_g.size(1))
        else:
            h_ud = torch.mm(a_ud, x)
            h = self.linear(x) + self.linear_ud(h_ud)
        h = F.relu(h)
        h = h.view(num_sent, sent_len, -1)
        if self.globalnode:
            return h, h_g
        return h
        # _h1 = self.sparse_mm(a1, x)
        # _h2 = self.sparse_mm(a2, x)


class GNN(nn.Module):

    def __init__(self, vocab_size, output_dim, args):
        super(GNN, self).__init__()
        # self.clstm = Character_LSTM(vocab_size, embed_dim, input_dim - feat_dim, dropout=dropout)
        self.char_cnn = Character_CNN(vocab_size, args.embed_dim, args.num_filters, args.filter_sizes)
        
        self.num_layers = len(args.graph_dim)
        self.lstm_layer = nn.ModuleList()
        self.gnn_layer = nn.ModuleList()
        in_dim = args.num_filters * len(args.filter_sizes) + args.feat_dim
        self.feat_dim = args.feat_dim
        for out_dim in args.graph_dim:
            if args.model in ['lstm', 'lstm+gnn']:
                self.lstm_layer.append(Word_LSTM(in_dim, out_dim))
                in_dim = out_dim
            if args.model in ['gnn', 'lstm+gnn']:
                self.gnn_layer.append(GNN_Layer(in_dim, out_dim, args.model, args.globalnode))
                in_dim = out_dim
        self.output = nn.Linear(args.graph_dim[-1], output_dim)

        self.globalnode = args.globalnode
        if self.globalnode:
            self.linear_global = nn.Linear(sum(args.graph_dim), args.num_forms)
        
        self.drop = nn.Dropout(p=args.dropout)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.model = args.model

    def forward(self, words, feature, mask, a_ud, a_lr):
        # h = self.clstm(words, length)
        h = self.char_cnn(words)
        h = self.drop(h)
        if self.feat_dim > 0:
            h = torch.cat((h, feature), 2)
        h_g = None
        for i in range(self.num_layers):
            if self.model in ['lstm', 'lstm+gnn']:
                h = self.lstm_layer[i](h)
            if self.model in ['gnn', 'lstm+gnn']:
                if self.globalnode:
                    h, g = self.gnn_layer[i](h, mask, a_ud, a_lr)
                    h_g = g if h_g is None else torch.cat((h_g, g), 1) 
                else:
                    h = self.gnn_layer[i](h, mask, a_ud, a_lr)
            h = self.drop(h)
        h = self.output(h)
        output = self.log_softmax(h)
        if self.globalnode:
            g = F.log_softmax(self.linear_global(h_g), dim=1)
            return output, g
        return output



