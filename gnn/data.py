import string
import os
import json
import torch
import numpy as np

vocab = ' ' + string.ascii_letters + string.digits + string.punctuation
vocab_size = len(vocab)
character2id = {c:idx for idx, c in enumerate(vocab)}

class DataPoint():

    def __init__(self, obj, graph_filename, label2id):

        self.set_id = obj['set_id']
        self.fax_id = obj['fax_id']

        # input_word = []
        words = []
        sents = []
        word_feat = []
        tags = []
        G = {'lr':[], 'ud':[]}
        for l in file(graph_filename):
            a = l.strip().split(' ')
            if l[0] != '#':
                label = a[0]
                if label not in label2id:
                    label2id[label] = len(label2id)
                tags.append(label2id[label])
                word = a[5]
                words.append(word)
                # input_word.append([character2id[c] for c in word if c in character2id])
                x, y = float(a[1]), float(a[2])
                w, h = float(a[3]) - x, float(a[4]) - y
                word_feat.append([x, y, w, h])
            elif a[0] == '#l':
                sents.append([int(x) for x in a[1:]])
                for i in range(1, len(a)-1):
                    G['lr'].append((int(a[i]), int(a[i+1]), 1.))
                    G['lr'].append((int(a[i+1]), int(a[i]), 1.))
            else:
                u,v,w = int(a[1]), int(a[2]), float(a[4])
                typ = a[3]
                G[typ].append((u,v,w))
                G[typ].append((v,u,w))
        
        self.max_word_len = max([len(w) for w in words])
        self.max_sent_len = max([len(s) for s in sents])
        self.num_sent = len(sents)
        self.num_word = len(words)
        self.num_feat = len(word_feat[0])
        self.sents = sents
        self.words = words
        data = torch.LongTensor(self.num_sent, self.max_sent_len, self.max_word_len).zero_()
        feat = torch.FloatTensor(self.num_sent, self.max_sent_len, self.num_feat).zero_()
        mask = torch.zeros(self.num_sent, self.max_sent_len)
        output = torch.LongTensor(self.num_sent, self.max_sent_len).zero_()
        word_id = {}
        for i in range(self.num_sent):
            for j in range(len(sents[i])):
                mask[i,j] = 1
                for k, c in enumerate(words[sents[i][j]]):
                    data[i,j,k] = character2id[c]
                feat[i,j] = torch.FloatTensor(word_feat[sents[i][j]])
                output[i,j] = tags[sents[i][j]]
                word_id[sents[i][j]] = i * self.max_sent_len + j
        #self.input = torch.nn.utils.rnn.pack_padded_sequence(Variable(data).cuda(), length, batch_first=True)
        self.input_word = data
        self.input_feat = feat
        self.output = output
        self.mask = mask

        # wordid2id = {order[i]:i for i in range(n)}
        # self.words = [words[i] for i in order]

        i = torch.LongTensor([
            [word_id[tp[1]] for tp in G['ud']],
            [word_id[tp[0]] for tp in G['ud']]
        ])
        v = torch.FloatTensor([1. for tp in G['ud']])
        # d = torch.zeros(self.num_word)
        # for tp in G['ud']:
        #     d[tp[1]] += 1
        # v = torch.FloatTensor([1./np.sqrt(d[tp[1]]*d[tp[0]]) for tp in G['ud']])
        n = self.num_sent * self.max_sent_len
        self.a_ud = torch.sparse.FloatTensor(i, v, torch.Size([n, n])).to_dense() # 

        i = torch.LongTensor([
            [word_id[tp[1]] for tp in G['lr']],
            [word_id[tp[0]] for tp in G['lr']]
        ])
        v = torch.FloatTensor([1. for tp in G['lr']])
        # d = torch.zeros(self.num_word)
        # for tp in G['lr']:
        #     d[tp[1]] += 1
        # v = torch.FloatTensor([1./np.sqrt(d[tp[1]]*d[tp[0]]) for tp in G['lr']])
        self.a_lr = torch.sparse.FloatTensor(i, v, torch.Size([n, n])).to_dense() # 

        self.valid = True 
        for label in label2id: # ['PtFN','PtLN']:#
            if label2id[label] not in tags:
                self.valid = False
                try:
                    print self.set_id, self.fax_id, label, obj['attrs']['Physician (Last Name,  First Name Mid Intial)']
                except:
                    pass



class DataSet():

    def __init__(self, data_path, graph_path):

        self.label2id = {}
        self.data = []
        self.vocab_size = vocab_size
        
        for filename in os.listdir(data_path):
            name, ext = os.path.splitext(filename)
            if ext != '.json':
                continue
            f = file(os.path.join(data_path, filename))
            for line in f:
                obj = json.loads(line)
                set_id = obj['set_id']
                fax_id = obj['fax_id']
                graph_filename = os.path.join(graph_path, str(set_id), str(fax_id) + '.txt')
                if not os.path.isfile(graph_filename):
                    continue
                # if set_id not in [14, 21]:
                #     continue
                x = DataPoint(obj, graph_filename, self.label2id)
                if x.valid:
                    self.data.append(x)
            # if set_id <= 20:
            #     train.extend(data)
            # elif set_id <= 20:
            #     train.extend(data[:len(data)/2]) 
            #     test.extend(data[len(data)/2:])
            # else:
            #     if set_id == 22:
            #         continue
            #     test.extend(data)

        self.train = self.data
        self.valid = []
        self.test = []

        self.order = None


    def split_train_valid_test(self, ratio, split, offset):
        n = len(self.data)
        if self.order == None:
            p = 9369319
            self.order = [i*p%n for i in range(n)]
        order = self.order[n*offset/split:n] + self.order[:n*offset/split]
        train_size = int(n*ratio[0])
        valid_size = int(n*ratio[1])
        self.train = [self.data[i] for i in order[:train_size]]
        self.valid = [self.data[i] for i in order[train_size:train_size+valid_size]]
        self.test = [self.data[i] for i in order[train_size+valid_size:]]

    def split_train_valid_test_byset(self, ratio, split, offset):
        n = 23
        order = range(n/split*offset, n) + range(n/split*offset)
        train_set = order[:int(n*ratio[0])]
        valid_set = order[int(n*ratio[0]):int(n*(ratio[0]+ratio[1]))]
        test_set = order[int(n*(ratio[0]+ratio[1])):]
        self.train = [x for x in self.data if x.set_id in train_set]
        self.valid = [x for x in self.data if x.set_id in valid_set]
        self.test = [x for x in self.data if x.set_id in test_set]
        print 'valid', list(valid_set)
        print 'test', list(test_set)



def read_data(data_path, graph_path):
    return DataSet(data_path, graph_path)




