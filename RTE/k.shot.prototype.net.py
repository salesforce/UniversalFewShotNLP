# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from scipy.stats import beta
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score

from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaModel#RobertaForSequenceClassification


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'

def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    print('starting model storing....')
    # model.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir)
    # tokenizer.save_pretrained(output_dir)
    print('store succeed')




class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id







def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask):
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)

        last_hidden, score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return last_hidden, score_single



class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        last_hidden = torch.tanh(x)
        x = self.dropout(last_hidden)
        x = self.out_proj(x)
        return last_hidden, x

class PrototypeNet(nn.Module):
    def __init__(self, hidden_size):
        super(PrototypeNet, self).__init__()
        self.HiddenLayer_1 = nn.Linear(3*hidden_size, 2*hidden_size)
        self.HiddenLayer_2 = nn.Linear(2*hidden_size, hidden_size)
        self.HiddenLayer_3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, rep_classes,rep_query_batch):
        '''
        rep_classes: (#class, hidden_size)
        rep_query_batch: (batch_size, hidden_size)
        '''
        class_size = rep_classes.shape[0]
        batch_size = rep_query_batch.shape[0]
        repeat_rep_classes = rep_classes.repeat(batch_size, 1)
        repeat_rep_query = torch.repeat_interleave(rep_query_batch, repeats=class_size, dim=0)
        combined_rep = torch.cat([repeat_rep_classes, repeat_rep_query, repeat_rep_classes*repeat_rep_query], dim=1) #(#class*batch, 3*hidden)

        all_scores = torch.sigmoid(self.HiddenLayer_3(self.dropout(torch.tanh(self.HiddenLayer_2(self.dropout(torch.tanh(self.HiddenLayer_1(combined_rep)))))))) #(#class*batch, 1)
        score_matrix = all_scores.view(-1, class_size) #(batch_size, class_size)
        return score_matrix


def get_RTE_as_train_k_shot(filename, k_shot):
    '''
    can read the training file, dev and test file
    '''
    examples_entail=[]
    examples_non_entail=[]
    readfile = codecs.open(filename, 'r', 'utf-8')
    line_co=0
    for row in readfile:
        if line_co>0:
            line=row.strip().split('\t')
            guid = "train-"+str(line_co-1)
            text_a = line[1].strip()
            text_b = line[2].strip()
            label = 'entailment' if line[3].strip()=='entailment' else 'not_entailment' #["entailment", "not_entailment"]
            if label == 'entailment':
                examples_entail.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            else:
                examples_non_entail.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        line_co+=1
    readfile.close()
    print('loaded  entail size:', len(examples_entail), 'non-entail size:', len(examples_non_entail))
    '''sampling'''
    if k_shot > 99999:
        return examples_entail, examples_non_entail
    else:
        return random.sample(examples_entail, k_shot), random.sample(examples_non_entail, k_shot)


def get_RTE_as_dev(filename):
    '''
    can read the training file, dev and test file
    '''
    examples=[]
    readfile = codecs.open(filename, 'r', 'utf-8')
    line_co=0
    for row in readfile:
        if line_co>0:
            line=row.strip().split('\t')
            guid = "dev-"+str(line_co-1)
            text_a = line[1].strip()
            text_b = line[2].strip()
            # label = line[3].strip() #["entailment", "not_entailment"]
            label = 'entailment' if line[3].strip()=='entailment' else 'not_entailment'
            # label = 'entailment'  if line[3] == 'entailment' else 'neutral'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        line_co+=1
        # if line_co > 20000:
        #     break
    readfile.close()
    print('loaded  size:', line_co-1)
    return examples

def get_RTE_as_test(filename):
    readfile = codecs.open(filename, 'r', 'utf-8')
    line_co=0
    examples=[]
    for row in readfile:
        line=row.strip().split('\t')
        if len(line)==3:
            guid = "test-"+str(line_co)
            text_a = line[1]
            text_b = line[2]
            '''for RTE, we currently only choose randomly two labels in the set, in prediction we then decide the predicted labels'''
            label = 'entailment'  if line[0] == '1' else 'not_entailment'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            line_co+=1

    readfile.close()
    print('loaded test size:', line_co)
    return examples

def get_MNLI_train(filename, k_shot):
    '''
    classes: ["entailment", "neutral", "contradiction"]
    '''
    examples_entail = []
    examples_neural = []
    examples_contra = []
    readfile = codecs.open(filename, 'r', 'utf-8')
    line_co=0
    for row in readfile:
        if line_co>0:
            line=row.strip().split('\t')
            guid = "train-"+str(line_co-1)
            # text_a = 'MNLI. '+line[8].strip()
            text_a = line[8].strip()
            text_b = line[9].strip()
            label = line[-1].strip() #["entailment", "neutral", "contradiction"]

            if label == 'entailment':
                examples_entail.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            elif label == 'neutral':
                examples_neural.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            else:
                examples_contra.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        line_co+=1
    readfile.close()
    print('loaded  MNLI size:', len(examples_entail)+len(examples_neural)+len(examples_contra))

    kshot_entail = random.sample(examples_entail, k_shot)
    kshot_neural = random.sample(examples_neural, k_shot)
    kshot_contra = random.sample(examples_contra, k_shot)

    remaining_examples = []
    for ex in examples_entail+examples_neural+examples_contra:
        if ex not in set(kshot_entail+kshot_neural+kshot_contra):
            remaining_examples.append(ex)

    assert len(kshot_entail)+len(kshot_neural)+len(kshot_contra)+len(remaining_examples)==len(examples_entail+examples_neural+examples_contra)
    return kshot_entail, kshot_neural, kshot_contra, remaining_examples


def examples_to_features(source_examples, label_list, args, tokenizer, batch_size, output_mode, dataloader_mode='sequential'):
    source_features = convert_examples_to_features(
        source_examples, label_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    dev_all_input_ids = torch.tensor([f.input_ids for f in source_features], dtype=torch.long)
    dev_all_input_mask = torch.tensor([f.input_mask for f in source_features], dtype=torch.long)
    dev_all_segment_ids = torch.tensor([f.segment_ids for f in source_features], dtype=torch.long)
    dev_all_label_ids = torch.tensor([f.label_id for f in source_features], dtype=torch.long)

    dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_label_ids)
    if dataloader_mode=='sequential':
        dev_sampler = SequentialSampler(dev_data)
    else:
        dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)


    return dev_dataloader



def main():
    parser = argparse.ArgumentParser()


    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument('--kshot',
                        type=int,
                        default=5,
                        help="random seed for initialization")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
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
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")


    args = parser.parse_args()



    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



    target_kshot_entail_examples, target_kshot_nonentail_examples = get_RTE_as_train_k_shot('/export/home/Dataset/glue_data/RTE/train.tsv', args.kshot) #train_pu_half_v1.txt
    target_dev_examples = get_RTE_as_dev('/export/home/Dataset/glue_data/RTE/dev.tsv')
    target_test_examples = get_RTE_as_test('/export/home/Dataset/RTE/test_RTE_1235.txt')
    source_kshot_entail, source_kshot_neural, source_kshot_contra, source_remaining_examples = get_MNLI_train('/export/home/Dataset/glue_data/MNLI/train.tsv', args.kshot)
    source_examples = source_kshot_entail+ source_kshot_neural+ source_kshot_contra+ source_remaining_examples
    target_label_list = ["entailment", "not_entailment"]
    source_label_list = ["entailment", "neutral", "contradiction"]
    source_num_labels = len(source_label_list)
    target_num_labels = len(target_label_list)
    print('training size:', len(source_examples), 'dev size:', len(target_dev_examples), 'test size:', len(target_test_examples))

    num_train_optimization_steps = None
    num_train_optimization_steps = int(
        len(source_remaining_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    roberta_model = RobertaForSequenceClassification(3)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    roberta_model.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/MNLI_pretrained/_acc_0.9040886899918633.pt'))
    roberta_model.to(device)
    roberta_model.eval()

    protonet = PrototypeNet(bert_hidden_dim)
    protonet.to(device)

    param_optimizer = list(protonet.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_acc = 0.0
    max_dev_acc = 0.0

    retrieve_batch_size = 5

    source_kshot_entail_dataloader = examples_to_features(source_kshot_entail, source_label_list, args, tokenizer, retrieve_batch_size, "classification", dataloader_mode='sequential')
    source_kshot_neural_dataloader = examples_to_features(source_kshot_neural, source_label_list, args, tokenizer, retrieve_batch_size, "classification", dataloader_mode='sequential')
    source_kshot_contra_dataloader = examples_to_features(source_kshot_contra, source_label_list, args, tokenizer, retrieve_batch_size, "classification", dataloader_mode='sequential')
    source_remain_ex_dataloader = examples_to_features(source_remaining_examples, source_label_list, args, tokenizer, args.train_batch_size, "classification", dataloader_mode='random')

    target_kshot_entail_dataloader = examples_to_features(target_kshot_entail_examples, target_label_list, args, tokenizer, retrieve_batch_size, "classification", dataloader_mode='sequential')
    target_kshot_nonentail_dataloader = examples_to_features(target_kshot_nonentail_examples, target_label_list, args, tokenizer, retrieve_batch_size, "classification", dataloader_mode='sequential')
    target_dev_dataloader = examples_to_features(target_dev_examples, target_label_list, args, tokenizer, args.eval_batch_size, "classification", dataloader_mode='random')
    target_test_dataloader = examples_to_features(target_test_examples, target_label_list, args, tokenizer, args.eval_batch_size, "classification", dataloader_mode='random')

    '''starting to train'''
    iter_co = 0
    final_test_performance = 0.0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(source_remain_ex_dataloader, desc="Iteration")):
            protonet.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids_batch = batch

            roberta_model.eval()
            with torch.no_grad():
                last_hidden_batch, _ = roberta_model(input_ids, input_mask)
            '''
            retrieve rep for support examples
            '''
            kshot_entail_reps = []
            for entail_batch in source_kshot_entail_dataloader:
                entail_batch = tuple(t.to(device) for t in entail_batch)
                input_ids, input_mask, segment_ids, label_ids = entail_batch
                roberta_model.eval()
                with torch.no_grad():
                    last_hidden_entail, _ = roberta_model(input_ids, input_mask)
                kshot_entail_reps.append(last_hidden_entail)
            kshot_entail_rep = torch.mean(torch.cat(kshot_entail_reps, dim=0), dim=0, keepdim=True)
            kshot_neural_reps = []
            for neural_batch in source_kshot_neural_dataloader:
                neural_batch = tuple(t.to(device) for t in neural_batch)
                input_ids, input_mask, segment_ids, label_ids = neural_batch
                roberta_model.eval()
                with torch.no_grad():
                    last_hidden_neural, _ = roberta_model(input_ids, input_mask)
                kshot_neural_reps.append(last_hidden_neural)
            kshot_neural_rep = torch.mean(torch.cat(kshot_neural_reps, dim=0), dim=0, keepdim=True)
            kshot_contra_reps = []
            for contra_batch in source_kshot_contra_dataloader:
                contra_batch = tuple(t.to(device) for t in contra_batch)
                input_ids, input_mask, segment_ids, label_ids = contra_batch
                roberta_model.eval()
                with torch.no_grad():
                    last_hidden_contra, _ = roberta_model(input_ids, input_mask)
                kshot_contra_reps.append(last_hidden_contra)
            kshot_contra_rep = torch.mean(torch.cat(kshot_contra_reps, dim=0), dim=0, keepdim=True)

            class_prototype_reps = torch.cat([kshot_entail_rep, kshot_neural_rep, kshot_contra_rep], dim=0) #(3, hidden)

            '''forward to model'''
            batch_logits = protonet(class_prototype_reps, last_hidden_batch)

            loss_fct = CrossEntropyLoss()

            loss = loss_fct(batch_logits.view(-1, source_num_labels), label_ids_batch.view(-1))

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            iter_co+=1
            # if iter_co %20==0:
            if iter_co % len(source_remain_ex_dataloader)==0:
                '''
                start evaluate on dev set after this epoch
                '''
                protonet.eval()
                '''first get representations for support examples'''
                kshot_entail_reps = []
                for entail_batch in target_kshot_entail_dataloader:
                    entail_batch = tuple(t.to(device) for t in entail_batch)
                    input_ids, input_mask, segment_ids, label_ids = entail_batch
                    roberta_model.eval()
                    with torch.no_grad():
                        last_hidden_entail, _ = roberta_model(input_ids, input_mask)
                    kshot_entail_reps.append(last_hidden_entail)
                kshot_entail_rep = torch.mean(torch.cat(kshot_entail_reps, dim=0), dim=0, keepdim=True)
                kshot_nonentail_reps = []
                for nonentail_batch in target_kshot_nonentail_dataloader:
                    nonentail_batch = tuple(t.to(device) for t in nonentail_batch)
                    input_ids, input_mask, segment_ids, label_ids = nonentail_batch
                    roberta_model.eval()
                    with torch.no_grad():
                        last_hidden_nonentail, _ = roberta_model(input_ids, input_mask)
                    kshot_nonentail_reps.append(last_hidden_nonentail)
                kshot_nonentail_rep = torch.mean(torch.cat(kshot_nonentail_reps, dim=0), dim=0, keepdim=True)
                target_class_prototype_reps = torch.cat([kshot_entail_rep, kshot_nonentail_rep], dim=0) #(2, hidden)

                for idd, dev_or_test_dataloader in enumerate([target_dev_dataloader, target_test_dataloader]):


                    eval_loss = 0
                    nb_eval_steps = 0
                    preds = []
                    gold_label_ids = []
                    # print('Evaluating...')
                    for input_ids, input_mask, segment_ids, label_ids in dev_or_test_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        gold_label_ids+=list(label_ids.detach().cpu().numpy())
                        roberta_model.eval()
                        with torch.no_grad():
                            last_hidden_target_batch, _ = roberta_model(input_ids, input_mask)

                        with torch.no_grad():
                            logits = protonet(target_class_prototype_reps, last_hidden_target_batch)


                        if len(preds) == 0:
                            preds.append(logits.detach().cpu().numpy())
                        else:
                            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                    preds = preds[0]

                    pred_probs = softmax(preds,axis=1)
                    pred_label_ids = list(np.argmax(pred_probs, axis=1))

                    gold_label_ids = gold_label_ids
                    assert len(pred_label_ids) == len(gold_label_ids)
                    hit_co = 0
                    for k in range(len(pred_label_ids)):
                        if pred_label_ids[k] == gold_label_ids[k]:
                            hit_co +=1
                    test_acc = hit_co/len(gold_label_ids)

                    if idd == 0: # this is dev
                        if test_acc > max_dev_acc:
                            max_dev_acc = test_acc
                            print('\ndev acc:', test_acc, ' max_dev_acc:', max_dev_acc, '\n')

                        else:
                            print('\ndev acc:', test_acc, ' max_dev_acc:', max_dev_acc, '\n')
                            break
                    else: # this is test
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc

                        final_test_performance = test_acc
                        print('\ntest acc:', test_acc, ' max_test_acc:', max_test_acc, '\n')
    print('final_test_performance:', final_test_performance)



if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=6 python -u k.shot.prototype.net.py --do_lower_case --num_train_epochs 20 --train_batch_size 32 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 128 --seed 42 --kshot 5

'''
