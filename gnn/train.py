import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from data import read_data
from gnn import GNN
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay (default: 0.001)')
parser.add_argument('--cuda', action='store_true', default=True, help='enable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--patience', type=int, default=3, help='number of times to observe worsening validation set error before giving up')
parser.add_argument('--embed_dim', type=int, default=64, help='character embedding dimension')
parser.add_argument('--filter_sizes', type=str, default='2,3,4', help='character cnn filter size')
parser.add_argument('--num_filters', type=int, default=64, help='character cnn filter number')
parser.add_argument('--feat_dim', type=int, default=4, help='additional feature')
parser.add_argument('--graph_dim', type=str, default='64,32,16', help='lstm and graph layer dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--model', type=str, default='gnn', choices=['gnn','lstm','lstm+gnn'], help='model')
parser.add_argument('--output', type=str, default='output', help='output name')
parser.add_argument('--globalnode', action='store_true', default=False, help='add global node')
parser.add_argument('--num_forms', type=int, default=23, help='number of form types')
args = parser.parse_args()

args.filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
args.graph_dim = [int(x) for x in args.graph_dim.split(',')]

print 'random seed:', args.seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# torch.backends.cudnn.enabled = False


dataset = read_data('../data', '../graph')
label2id = dataset.label2id
print label2id

vocab_size = dataset.vocab_size
output_dim = len(label2id)

def acc_to_str(acc):
    s = ['%s:%.3f'%(label, acc[label]) for label in acc]
    return '{' + ', '.join(s) + '}'

cross_res = {label:[] for label in label2id if label != 'O'}
output_file = file('%s.mistakes' % args.output, 'w')

for cross_valid in range(5):

    model = GNN(vocab_size=vocab_size, output_dim=output_dim, args=args)
    model.cuda()
    # print vocab_size

    dataset.split_train_valid_test([0.8, 0.1, 0.1], 5, cross_valid)
    print 'train:', len(dataset.train), 'valid:', len(dataset.valid), 'test:', len(dataset.test)

    def evaluate(model, datalist, output_file=None):
        if output_file != None:
            output_file.write('#############################################\n')
        correct = {label:0 for label in label2id if label != 'O'}
        total = len(datalist)
        model.eval()
        print_cnt = 0
        for data in datalist:
            word, feat = Variable(data.input_word).cuda(), Variable(data.input_feat).cuda()
            a_ud, a_lr = Variable(data.a_ud, requires_grad=False).cuda(), Variable(data.a_lr, requires_grad=False).cuda()
            mask = Variable(data.mask, requires_grad=False).cuda()
            if args.globalnode:
                logprob, form = model(word, feat, mask, a_ud, a_lr)
                logprob = logprob.data.view(-1, output_dim)
            else:
                logprob = model(word, feat, mask, a_ud, a_lr).data.view(-1, output_dim)
            mask = mask.data.view(-1)
            y_pred = torch.LongTensor(output_dim)
            for i in range(output_dim):
                prob = logprob[:,i].exp() * mask
                y_pred[i] = prob.topk(k=1)[1][0]
            # y_pred = logprob.topk(k=1,dim=0)[1].view(-1)
            for label in label2id:
                if label == 'O':
                    continue
                labelid = label2id[label]
                if data.output.view(-1)[y_pred[labelid]] == labelid:
                    correct[label] += 1
                else:
                    if output_file != None:
                        num_sent, sent_len, word_len = data.input_word.size()
                        id = y_pred[label2id[label]]
                        word = data.words[data.sents[id/sent_len][id%sent_len]]
                        output_file.write('%d %d %s %s\n' % (data.set_id, data.fax_id, label, word))
                    # if print_err and label == 'FN' and print_cnt < 1:
                    #     num_sent, sent_len, word_len = data.input_word.size()
                    #     id = y_pred[label2id['FN']]
                    #     try:
                    #         print data.set_id, data.fax_id, data.words[data.sents[id/sent_len][id%sent_len]], list(logprob[id])
                    #         print_cnt += 1
                    #     except:
                    #         continue

        return {label:float(correct[label])/total for label in correct}

    batch = 1

    weight = torch.zeros(len(label2id))
    for label, id in label2id.items():
        weight[id] = 1 if label == 'O' else 10
    loss_function = nn.NLLLoss(weight.cuda(), reduce=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/float(batch), weight_decay=args.wd)

    best_acc = -1
    wait = 0

    for epoch in range(args.epochs):
        sum_loss = 0
        model.train()
        for idx, data in enumerate(dataset.train):
            word, feat = Variable(data.input_word).cuda(), Variable(data.input_feat).cuda()
            a_ud, a_lr = Variable(data.a_ud, requires_grad=False).cuda(), Variable(data.a_lr, requires_grad=False).cuda()
            mask = Variable(data.mask, requires_grad=False).cuda()
            true_output = Variable(data.output).cuda()
            if args.globalnode:
                logprob, form = model(word, feat, mask, a_ud, a_lr)
            else:
                logprob = model(word, feat, mask, a_ud, a_lr)
            loss = torch.mean(mask.view(-1) * loss_function(logprob.view(-1, output_dim), true_output.view(-1)))
            if args.globalnode:
                true_form = Variable(torch.LongTensor([data.set_id - 1])).cuda()
                loss = loss + 0.1 * F.nll_loss(form, true_form)
            sum_loss += loss.data.sum()
            loss.backward()
            if (idx + 1) % batch == 0 or idx + 1 == len(dataset.train):
                optimizer.step()
                optimizer.zero_grad()
        train_acc = evaluate(model, dataset.train)
        valid_acc = evaluate(model, dataset.valid)
        test_acc = evaluate(model, dataset.test)
        print 'Epoch %d:  Train Loss: %.3f  Train: %s  Valid: %s  Test: %s' \
            % (epoch, sum_loss, acc_to_str(train_acc), acc_to_str(valid_acc), acc_to_str(test_acc))

        if epoch < 5:
            continue
        acc = sum(valid_acc.values())
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.output+'.model')
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            break

    model.load_state_dict(torch.load(args.output+'.model'))
    test_acc = evaluate(model, dataset.test, output_file=output_file)
    print '########', acc_to_str(test_acc)
    for label in test_acc:
        cross_res[label].append(test_acc[label])

print "Cross Validation Result:"
for label in cross_res:
    print label, np.mean(cross_res[label])

