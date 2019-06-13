import argparse
import os
import json
import itertools
from itertools import product, permutations
from random import sample

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertForMaskedLM, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from run_child_finetuning import *
#from child_frames import frames
#from child_wsc_generator import make_sentences
from child_generator import make_sentences

BERT_DIR = '/nas/pretrain-bert/pretrain-pytorch/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained('/nas/pretrain-bert/pretrain-pytorch/bert-base-uncased-vocab.txt')


parser = argparse.ArgumentParser()

parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=3e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumualte before performing a backward/update pass.")
parser.add_argument("--dev_percent",
                    default=0.5,
                    type=float)
# args = parser.parse_args(['--output_dir', '/home'])
# args = parser.parse_args([])
args = parser.parse_args()
args.do_lower_case = True
args.do_train = True
args.do_eval = True
args.eval_batch_size = 128
# args.learning_rate = 1e-4
#args.num_train_epochs = 100
print(args)

sentences = make_sentences(maybe=False, structured=False)
#sentences = []
#for frame in frames:
#    sentences += make_sentences(**frame)[-1]
logger.info('num_sent = %d' % len(sentences))
child_dataset = CHILDDataset(tokenizer, sentences, dev_percent=args.dev_percent)
train_features = child_dataset.get_train_features()
logger.info('num_train_examples = %d' % len(train_features))
num_train_steps = int(
    len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
logger.info('num_train_steps = %d' % num_train_steps)
eval_features = child_dataset.get_dev_features()

train_dataset = child_dataset.build_dataset(train_features)
eval_dataset = child_dataset.build_dataset(eval_features)

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))

args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

# Prepare model
model = BertForMaskedLM.from_pretrained(BERT_DIR)
#CONFIG_NAME = 'bert_config_small.json'
#config = BertConfig(os.path.join(BERT_DIR, CONFIG_NAME))
#model = BertForMaskedLM(config)
_ = model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args.learning_rate,
                     warmup=args.warmup_proportion,
                     t_total=num_train_steps)

logger.info("Epoch 0")
logger.info("Evaluating on train set...")
#validate(model, train_dataset, device)
logger.info("Evaluating on valid set...")
#validate(model, eval_dataset, device)

global_step = 0
for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
    _ = model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
#     for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch_idx in enumerate(get_batch_index(len(train_dataset), args.train_batch_size, randomized=True)):
        batch = tuple(t[batch_idx] for t in train_dataset.tensors)
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
        loss = model(input_ids, segment_ids, input_mask, lm_label_ids)
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
            if global_step % 1000 == 0:
                print('global_step %d, lr = %f' % (global_step, lr_this_step))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    if args.do_eval:
        logger.info("Epoch %d" % (epoch + 1))
        logger.info("Evaluating on train set...")
        validate(model, train_dataset, device)
        logger.info("Evaluating on valid set...")
        validate(model, eval_dataset, device)
