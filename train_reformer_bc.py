import h5py
import numpy as np
import argparse
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import random
import shutil
import time
import warnings
import math
import Bio.Seq
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
import torch.utils.data
import transformers as T
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j',
                    '--workers',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs',
                    default=6,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=2e-5,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-5,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument("--lr_scheduler_type", type=str,
                    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                    default="cosine", help="The scheduler type to use.")
parser.add_argument('--outdir', help='output directory')
parser.add_argument('--h5file', help='input h5file')
parser.add_argument('--device', nargs='+', help='a list of gpu')
parser.add_argument('--resume', help='resume from checkpoint')

'''
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, tokenizer, max_length=512, train=False):
        df = h5py.File(h5file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if train:
            self.sequence = df['trn_seq']
            self.label = df['trn_label']
            self.barcode = df['trn_code_prefix']
        else:
            self.sequence = df['val_seq']
            self.label = df['val_label']
            self.barcode = df['val_code_prefix']

        self.n = len(self.label)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ss = self.sequence[i].decode()
        label = int(self.label[i])
        ss = [ss[i:int(i+3)] for i in range(int(len(ss)-2))]# 3 mer data
        seq = [self.barcode[i].decode()]
        seq.extend(ss[:-1])
        inputs = self.tokenizer(seq, is_split_into_words=True, add_special_tokens=True, return_tensors='pt') 
        label = torch.tensor(label)
        return inputs['input_ids'], label
'''
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, tokenizer, max_length=512, mode="train"):
        df = h5py.File(h5file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if mode == "train":
            self.sequence = df['trn_seq']      
            self.label = df['trn_label']
            self.barcode = df['trn_code_prefix']
        elif mode == "val":
            self.sequence = df['val_seq']    
            self.label = df['val_label']
            self.barcode = df['val_code_prefix']
        elif mode == "test":
            self.sequence = df['test_seq']  # Ensure 'test_seq' exists in HDF5 file
            self.barcode = df['test_code_prefix']
            self.label = None
        else:
            raise ValueError("Invalid mode! Choose from ['train', 'val', 'test'].")

        self.n = len(self.sequence)
    
    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ss = self.sequence[i].decode()
        #label = int(self.label[i])
        ss = [ss[i:int(i+3)] for i in range(int(len(ss)-2))]# 3 mer data
        seq = [self.barcode[i].decode()]
        seq.extend(ss[:-1])
        inputs = self.tokenizer(seq, is_split_into_words=True, add_special_tokens=True, return_tensors='pt') 
        
        if self.label is not None:
            label = int(self.label[i])
            label = torch.tensor(label)
            return inputs['input_ids'], label
        else:
            return inputs['input_ids']  # No label in test mode

class Bert4BinaryClassification(nn.Module):
    def __init__(self,tokenizer):
        super(Bert4BinaryClassification, self).__init__()
        self.model = T.BertModel.from_pretrained("armheb/DNA_bert_3")
        self.model.resize_token_embeddings(292)#(len(tokenizer))
        hidden_size = 768 #self.opt_model.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.lin = nn.Linear(hidden_size, 1, bias=False)
        self.classification = nn.Linear(509, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids):#, coverage=None):
        hidden = self.model(input_ids=input_ids.squeeze(1)).last_hidden_state[:,2:-1,:]
        hidden = self.dropout(hidden)
        score = self.lin(hidden).squeeze()
        predict = self.classification(score)
        predict = self.sigmoid(predict)
        return predict

def main():
    args = parser.parse_args()
    assert args.outdir is not None

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    gpus = args.device
    main_worker(gpus=gpus, args=args)

def main_worker(gpus, args):
    tokenizer = T.BertTokenizer.from_pretrained("armheb/DNA_bert_3")
    
    df = h5py.File(args.h5file)
    prefix_token = list(set([i.decode() for i in df['trn_code_prefix'][:]]))
    tokenizer.add_tokens(prefix_token)
    tokenizer.save_pretrained(args.outdir)
    model = Bert4BinaryClassification(tokenizer)
    df.close()
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location = 'cpu')
        model.load_state_dict(checkpoint)
    
    print(model.model.config)
    print(model)

    model.cuda()
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.95))

    criterion = nn.BCELoss()

    cudnn.benchmark = True
    #train_dataset = SequenceDataset(args.h5file, tokenizer, train=True)
    #val_dataset = SequenceDataset(args.h5file, tokenizer, train=False)
    train_dataset = SequenceDataset(args.h5file, tokenizer, train=True)
    val_dataset = SequenceDataset(args.h5file, tokenizer, train=False)

    PAD_TOKEN_ID = tokenizer.pad_token_id
    def collate_fn(x):
        input_ids = [ids.squeeze() for ids, _ in x]
        labels = [label for _, label in x]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, padding_value=PAD_TOKEN_ID)
        mask = (input_ids != 0).int()
        labels = torch.stack(labels)
        #return input_ids.T, mask.T, labels
        return {'input_ids':input_ids.T, 'attention_mask':mask.T, 'labels':labels}

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

  
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, shuffle=False, num_workers=1,
                                             pin_memory=True)
    num_update_steps_per_epoch = math.ceil(len(train_loader))
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=1e4, num_training_steps=max_train_steps)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_auc = 0.5
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, lr_scheduler, optimizer, epoch, args)

        # evaluate on validation set
        auc_score = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = auc_score > best_auc
        best_auc = max(auc_score, best_auc)

        #model.module.save_pretrained(args.outdir)
        torch.save(model.module.state_dict(), f'{args.outdir}/model_epoch{epoch}.bin')
        if is_best:
            torch.save(model.module.state_dict(), f'{args.outdir}/model_best.bin')
            
def train(train_loader, model, criterion, scheduler, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')
    aucs = AverageMeter('AUC', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, accs, aucs],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_ids, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_ids = input_ids.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True).view(-1).float()
        
        logits = model(input_ids).view(-1)
        loss = criterion(logits, label)

        # measure accuracy and record loss
        acc, auc = compute_metrics(logits.detach().cpu().squeeze(), label.detach().cpu().squeeze())
        losses.update(loss.item(), input_ids.size(0))
        
        accs.update(acc, input_ids.size(0))
        aucs.update(auc, input_ids.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')
    aucs = AverageMeter('AUC', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, accs, aucs],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_ids, label) in enumerate(val_loader):
            input_ids = input_ids.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True).view(-1).float()
            logits = model(input_ids).view(-1)
            loss = criterion(logits, label)

            # measure accuracy and record loss
            acc, auc = compute_metrics(logits.detach().cpu().squeeze(), label.detach().cpu().squeeze())
            losses.update(loss.item(), input_ids.size(0))
            accs.update(acc, input_ids.size(0))
            aucs.update(auc, input_ids.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
          
        # TODO: this should also be done with the ProgressMeter
        print(f' * Loss {losses.avg:.3f}')

    return aucs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def compute_metrics(outputs, labels):
    predicted_labels = (outputs >= 0.5).float()
    accuracy = accuracy_score(labels.cpu(), predicted_labels.cpu())    
    auc = roc_auc_score(labels.cpu(), outputs.cpu())
    return accuracy, auc

if __name__ == '__main__':
    main()
