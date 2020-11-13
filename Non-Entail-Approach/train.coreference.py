# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

# from preprocess import load_GAP_coreference_data
from gap_scorer_modified import run_scorer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.hidden_layer_0 = nn.Linear(bert_hidden_dim*3, bert_hidden_dim)
        self.hidden_layer_1 = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.hidden_layer_2 = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    # def forward(self, input_ids, input_mask):
    #     outputs_single = self.roberta_single(input_ids, input_mask, None)
    #     hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)
    #
    #     score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
    #     return score_single

    def forward(self, input_ids, input_mask, span_a_mask, span_b_mask):
        # single_train_input_ids, single_train_input_mask, single_train_segment_ids, single_train_label_ids = batch_single
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        output_last_layer_tensor3 = outputs_single[0] #(batch_size, sequence_length, hidden_size)`)
        span_a_reps = torch.sum(output_last_layer_tensor3*span_a_mask.unsqueeze(2), dim=1) #(batch, hidden)
        span_b_reps = torch.sum(output_last_layer_tensor3*span_b_mask.unsqueeze(2), dim=1) #(batch, hidden)
        combined_rep = torch.cat([span_a_reps, span_b_reps, span_a_reps*span_b_reps],dim=1) #(batch, 3*hidden)
        MLP_input = torch.tanh(self.hidden_layer_0(combined_rep))#(batch, hidden)

        hidden_states_single = torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(MLP_input)))) #(batch, hidden)

        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single


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
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, span_a_left=None, span_a_right=None, span_b_left=None, span_b_right=None, label=None, entity_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.span_a_left  = span_a_left
        self.span_a_right = span_a_right
        self.span_b_left = span_b_left
        self.span_b_right = span_b_right
        self.text_b = text_b
        self.label = label
        self.entity_label = entity_label

        # InputExample(guid=idd, text_a=premise, span_a_left=span_a_left, span_a_right=span_a_right, text_b=None, span_b_left=span_b_left, span_b_right=span_b_right, label=label, entity_label='A-coref'))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, id, input_ids, input_mask, segment_ids, span_a_mask, span_b_mask, label_id, entity_label_id):
        self.id = id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.span_a_mask = span_a_mask
        self.span_b_mask = span_b_mask
        self.entity_label_id = entity_label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""


    def get_GAP_coreference(self, filename, k_shot):
        '''
        selected_examples: [(idd, premise, hypy_A, entity_A_label, hypy_B, entity_B_label)]
        '''
        examples=[]
        selected_examples = self.load_GAP_coreference_data(filename, k_shot)
        two_false_size = 0
        for example in selected_examples:
            #(idd, premise, pronoun_pos_pair, ent_B_pos_pair, entity_B_label, 'B-coref)
            idd = int(example[0].split('-')[1])
            premise = example[1]
            span_a_left = example[2][0]
            span_a_right = example[2][1]

            span_b_left = example[3][0]
            span_b_right = example[3][1]

            type = example[5]

            label = 'entailment' if example[4]=='TRUE' else 'not_entailment' # 'TRUE' or 'FALSE'

            examples.append(
                InputExample(guid=idd, text_a=premise, span_a_left=span_a_left, span_a_right=span_a_right, text_b=None, span_b_left=span_b_left, span_b_right=span_b_right, label=label, entity_label=type))


            # examples.append(
            #     InputExample(guid=idd, text_a=premise, text_b=hypo_a, label= hypo_a_label, entity_label='A-coref'))
            # examples.append(
            #     InputExample(guid=idd, text_a=premise, text_b=hypo_b, label= hypo_b_label, entity_label='B-coref'))
        print('loaded test size:', len(examples))
        return examples

    def load_GAP_coreference_data(self, filename, k_shot):
        path = '/export/home/Dataset/gap_coreference/'

        def generate_word_level_range(sentence, pronoun_str, pronoun_position, entity_str, entity_position):
            pronoun_len = len(pronoun_str)
            entity_len = len(entity_str)
            # print(sentence, type(pronoun_position), type(pronoun_len), type(pronoun_str))
            assert sentence[pronoun_position: (pronoun_position+pronoun_len)] == pronoun_str
            assert sentence[entity_position: (entity_position+entity_len)] == entity_str

            pronoun_left_context = sentence[:pronoun_position]
            pronoun_word_size = len(pronoun_str.strip().split())
            pronoun_left_size = len(pronoun_left_context.strip().split())
            pronoun_right_size = pronoun_left_size+pronoun_word_size-1

            entity_left_context = sentence[:entity_position]
            entity_word_size = len(entity_str.strip().split())
            entity_left_size = len(entity_left_context.strip().split())
            entity_right_size = entity_left_size+entity_word_size-1

            return (pronoun_left_size, pronoun_right_size), (entity_left_size, entity_right_size)


        all_examples = []
        with open(path+filename) as tsvfile:
            reader = csv.DictReader(tsvfile, dialect='excel-tab')
            for row in reader:
                all_examples.append(row)

        '''select k examples'''
        if k_shot < 1.0:
            selected_examples = random.sample(all_examples, int(len(all_examples)*k_shot))
        else:
            selected_examples = all_examples

        print('read selected example size:', len(selected_examples))
        selected_example_list = []
        for example in selected_examples:
            idd = example['ID']
            premise = example['Text']
            pronoun = example['Pronoun']
            pronoun_pos = int(example['Pronoun-offset'])
            entity_A = example['A']
            entity_A_pos = int(example['A-offset'])
            entity_A_label = example['A-coref']
            pronoun_pos_pair, ent_A_pos_pair = generate_word_level_range(premise, pronoun, pronoun_pos, entity_A, entity_A_pos)
            selected_example_list.append((idd, premise, pronoun_pos_pair, ent_A_pos_pair, entity_A_label, 'A-coref'))

            entity_B = example['B']
            entity_B_pos = int(example['B-offset'])
            entity_B_label = example['B-coref']
            pronoun_pos_pair, ent_B_pos_pair = generate_word_level_range(premise, pronoun, pronoun_pos, entity_B, entity_B_pos)

            selected_example_list.append((idd, premise, pronoun_pos_pair, ent_B_pos_pair, entity_B_label, 'B-coref'))

        return selected_example_list


    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "not_entailment"]
        # return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def wordpairID_2_tokenpairID(sentence, wordindex_left, wordindex_right, full_token_id_list, tokenizer, sent_1=True):
    '''pls note that the input indices pair include the b in (a,b), but the output doesn't'''
    '''first find the position of [2,2]'''
    position_two_two = 0
    for i in range(len(full_token_id_list)):
        if full_token_id_list[i]==2 and full_token_id_list[i+1]==2:
            position_two_two = i
            break
    span = ' '.join(sentence.split()[wordindex_left: wordindex_right+1])
    if wordindex_left!=0:
        '''this span is the begining of the sent'''
        span=' '+span

    span_token_list = tokenizer.tokenize(span)
    span_id_list = tokenizer.convert_tokens_to_ids(span_token_list)
    # print('span:', span, 'span_id_list:', span_id_list)
    if sent_1:
        # for i in range(wordindex_left, len(full_token_id_list)-len(span_id_list)):
        for i in range(wordindex_left, position_two_two):
            if full_token_id_list[i:i+len(span_id_list)] == span_id_list:
                return i, i+len(span_id_list), span_token_list

        # for i in range(wordindex_left, position_two_two):
        #     if full_token_id_list[i] == span_id_list[0]:
        #         return i, i+len(span_id_list), span_token_list



        return None, None, span_token_list
    else:
        # print('position_two_two:', position_two_two)
        for i in range(position_two_two+2, len(full_token_id_list)):
            if full_token_id_list[i:i+len(span_id_list)] == span_id_list:
                return i, i+len(span_id_list), span_token_list

        return None, None, span_token_list

def convert_examples_to_features(examples, label_list, entity_label_list, max_seq_length,
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
    entity_label_map = {label : i for i, label in enumerate(entity_label_list)}

    features = []
    give_up = 0
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

        span_a_left, span_a_right, span_a_token_list = wordpairID_2_tokenpairID(example.text_a, example.span_a_left, example.span_a_right, input_ids, tokenizer, sent_1=True)
        span_b_left, span_b_right, span_b_token_list = wordpairID_2_tokenpairID(example.text_a, example.span_b_left, example.span_b_right, input_ids, tokenizer, sent_1=True)
        # print('span_b_left, span_b_right, span_b_token_list:', span_b_left, span_b_right, span_b_token_list)
        if span_a_left is None or span_b_left is None:
            '''give up this pair'''
            give_up+=1
            continue
        else:
            span_a_mask = [0]*len(input_ids)
            for i in range(span_a_left, span_a_right):
                span_a_mask[i]=1
            span_b_mask = [0]*len(input_ids)
            for i in range(span_b_left, span_b_right):
                span_b_mask[i]=1

        features.append(
                InputFeatures(id = example.guid,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              span_a_mask = span_a_mask,
                              span_b_mask = span_b_mask,
                              label_id=label_id,
                              entity_label_id = entity_label_map[example.entity_label]))
    print('give_up:', give_up)
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





def examples_to_features(source_examples, label_list, entity_label_list, args, tokenizer, batch_size, output_mode, dataloader_mode='sequential'):
    source_features = convert_examples_to_features(
        source_examples, label_list, entity_label_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    dev_all_idd = torch.tensor([f.id for f in source_features], dtype=torch.long)
    dev_all_input_ids = torch.tensor([f.input_ids for f in source_features], dtype=torch.long)
    dev_all_input_mask = torch.tensor([f.input_mask for f in source_features], dtype=torch.long)
    dev_all_segment_ids = torch.tensor([f.segment_ids for f in source_features], dtype=torch.long)
    dev_all_label_ids = torch.tensor([f.label_id for f in source_features], dtype=torch.long)
    dev_all_span_a_mask = torch.tensor([f.span_a_mask for f in source_features], dtype=torch.float)
    dev_all_span_b_mask = torch.tensor([f.span_b_mask for f in source_features], dtype=torch.float)

    dev_all_entity_label_ids = torch.tensor([f.entity_label_id for f in source_features], dtype=torch.long)

    dev_data = TensorDataset(dev_all_idd, dev_all_input_ids, dev_all_input_mask, dev_all_span_a_mask, dev_all_span_b_mask, dev_all_segment_ids, dev_all_label_ids, dev_all_entity_label_ids)
    if dataloader_mode=='sequential':
        dev_sampler = SequentialSampler(dev_data)
    else:
        dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)


    return dev_dataloader


def build_GAP_output_format(example_id_list, gold_label_ids, pred_prob_entail, pred_label_ids_3way, threshold, dev_or_test='dev'):
    id2labellist = {}
    id2scorelist = {}
    for ex_id, type, prob, entail_or_not in zip(example_id_list, gold_label_ids, pred_prob_entail, pred_label_ids_3way):
        labellist = id2labellist.get(ex_id)
        scorelist = id2scorelist.get(ex_id)
        if scorelist is None:
            scorelist = [0.0, 0.0]
        scorelist[type] = prob
        if labellist is None:
            labellist = ['', '']
        labellist[type] = True if prob > threshold else False

        id2labellist[ex_id] = labellist
        id2scorelist[ex_id] = scorelist
    '''remove conflict'''
    eval_output_list = []
    prefix = dev_or_test+'-'#'test-' #'test-'
    for ex_id, labellist in id2labellist.items():
        if labellist[0] is True and labellist[1] is True:
            scorelist = id2scorelist.get(ex_id)
            if scorelist[0] > scorelist[1]:
                eval_output_list.append([prefix+str(ex_id), True, False])
            else:
                eval_output_list.append([prefix+str(ex_id), False, True])
        else:
            eval_output_list.append([prefix+str(ex_id)]+labellist)
    return eval_output_list

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
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
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument('--kshot',
                        type=float,
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


    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

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

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))



    processor = processors[task_name]()
    output_mode = output_modes[task_name]


    train_examples = processor.get_GAP_coreference('gap-development.tsv', args.kshot) #train_pu_half_v1.txt
    dev_examples = processor.get_GAP_coreference('gap-validation.tsv', 1.0)
    test_examples = processor.get_GAP_coreference('gap-test.tsv', 1.0)
    label_list = ["entailment", "not_entailment"]
    entity_label_list = ["A-coref", "B-coref"]
    # train_examples = get_data_hulu_fewshot('train', 5)
    # train_examples, dev_examples, test_examples, label_list = load_CLINC150_with_specific_domain_sequence(args.DomainName, args.kshot, augment=False)
    num_labels = len(label_list)
    print('num_labels:', num_labels, 'training size:', len(train_examples), 'dev size:', len(dev_examples), 'test size:', len(test_examples))

    num_train_optimization_steps = None
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    model = RobertaForSequenceClassification(num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.to(device)

    param_optimizer = list(model.named_parameters())
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
    max_dev_threshold = 0.0
    if args.do_train:
        train_dataloader = examples_to_features(train_examples, label_list, entity_label_list, args, tokenizer, args.train_batch_size, "classification", dataloader_mode='random')
        dev_dataloader = examples_to_features(dev_examples, label_list, entity_label_list, args, tokenizer, args.eval_batch_size, "classification", dataloader_mode='sequential')
        test_dataloader = examples_to_features(test_examples, label_list, entity_label_list, args, tokenizer, args.eval_batch_size, "classification", dataloader_mode='sequential')

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)

        iter_co = 0
        final_test_performance = 0.0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_example_ids, input_ids, input_mask, span_a_mask, span_b_mask, segment_ids, label_ids, entity_label_ids = batch


                logits = model(input_ids, input_mask, span_a_mask, span_b_mask)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

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
                # if iter_co %100==0:
                #     print('iter_co:', iter_co, ' mean loss:', tr_loss/iter_co)
                if iter_co % len(train_dataloader)==0:

                    model.eval()

                    '''
                     dev set after this epoch
                    '''

                    logger.info("***** Running dev *****")
                    logger.info("  Num examples = %d", len(dev_examples))

                    eval_loss = 0
                    nb_eval_steps = 0
                    preds = []
                    gold_label_ids = []
                    example_id_list = []
                    for _, batch in enumerate(tqdm(dev_dataloader, desc="dev")):
                        input_indices, input_ids, input_mask, span_a_mask, span_b_mask, segment_ids, _, label_ids = batch
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        span_a_mask = span_a_mask.to(device)
                        span_b_mask = span_b_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        example_ids = list(input_indices.numpy())
                        example_id_list+=example_ids
                        gold_label_ids+=list(label_ids.detach().cpu().numpy())

                        with torch.no_grad():
                            logits = model(input_ids, input_mask, span_a_mask, span_b_mask)
                        if len(preds) == 0:
                            preds.append(logits.detach().cpu().numpy())
                        else:
                            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                    preds = preds[0]

                    pred_probs = softmax(preds,axis=1)
                    pred_label_ids_3way = list(np.argmax(pred_probs, axis=1))
                    pred_prob_entail = list(pred_probs[:,0])

                    assert len(example_id_list) == len(pred_prob_entail)
                    assert len(example_id_list) == len(gold_label_ids)
                    assert len(example_id_list) == len(pred_label_ids_3way)

                    best_current_dev_acc = 0.0
                    best_current_threshold = -10.0
                    for threshold in np.arange(0.99, 0.0, -0.01):
                        # print('example_id_list:', example_id_list)
                        eval_output_list = build_GAP_output_format(example_id_list, gold_label_ids, pred_prob_entail, pred_label_ids_3way, threshold, dev_or_test='validation')
                        dev_acc = run_scorer('/export/home/Dataset/gap_coreference/gap-validation.tsv', eval_output_list)
                        # print('dev_acc:', dev_acc)
                        # exit(0)
                        if dev_acc > best_current_dev_acc:
                            best_current_dev_acc = dev_acc
                            best_current_threshold = threshold
                    print('best_current_dev_threshold:', best_current_threshold, 'best_current_dev_acc:', best_current_dev_acc)

                    if best_current_dev_acc > max_dev_acc:
                        max_dev_acc = best_current_dev_acc
                        max_dev_threshold = best_current_threshold

                        '''eval on test set'''
                        logger.info("***** Running test *****")
                        logger.info("  Num examples = %d", len(test_examples))

                        eval_loss = 0
                        nb_eval_steps = 0
                        preds = []
                        gold_label_ids = []
                        example_id_list = []
                        for _, batch in enumerate(tqdm(test_dataloader, desc="test")):
                            input_indices, input_ids, input_mask, span_a_mask, span_b_mask, segment_ids, _, label_ids = batch
                            input_ids = input_ids.to(device)
                            input_mask = input_mask.to(device)
                            span_a_mask = span_a_mask.to(device)
                            span_b_mask = span_b_mask.to(device)
                            segment_ids = segment_ids.to(device)
                            label_ids = label_ids.to(device)
                            example_ids = list(input_indices.numpy())
                            example_id_list+=example_ids
                            gold_label_ids+=list(label_ids.detach().cpu().numpy())

                            with torch.no_grad():
                                logits = model(input_ids, input_mask, span_a_mask, span_b_mask)
                            if len(preds) == 0:
                                preds.append(logits.detach().cpu().numpy())
                            else:
                                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                        preds = preds[0]

                        pred_probs = softmax(preds,axis=1)
                        pred_label_ids_3way = list(np.argmax(pred_probs, axis=1))
                        pred_prob_entail = list(pred_probs[:,0])

                        assert len(example_id_list) == len(pred_prob_entail)
                        assert len(example_id_list) == len(gold_label_ids)
                        assert len(example_id_list) == len(pred_label_ids_3way)

                        threshold = max_dev_threshold
                        eval_output_list = build_GAP_output_format(example_id_list, gold_label_ids, pred_prob_entail, pred_label_ids_3way, threshold, dev_or_test='test')

                        test_acc = run_scorer('/export/home/Dataset/gap_coreference/gap-test.tsv', eval_output_list)
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc
                        print('current_test_acc:', test_acc, ' max_test_acc:', max_test_acc)
                        final_test_performance = test_acc
        print('final_test_performance:', final_test_performance)







if __name__ == "__main__":
    main()

'''
full-shot command:
92.24   89.93   90.18   CUDA_VISIBLE_DEVICES=7 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 1.0
90.84   90.00   89.80   CUDA_VISIBLE_DEVICES=6 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 0.9
88.78   89.34   89.17   CUDA_VISIBLE_DEVICES=5 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 0.8
85.57   89.15   88.82   CUDA_VISIBLE_DEVICES=4 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 0.7
82.28   88.55   87.42   CUDA_VISIBLE_DEVICES=3 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 0.6
86.67   89.16   82.69   CUDA_VISIBLE_DEVICES=2 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 0.5
85.13   87.56   67.50   CUDA_VISIBLE_DEVICES=1 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 0.4
80.34   84.93   86.02   CUDA_VISIBLE_DEVICES=0 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 0.3
73.75   70.83   78.59   CUDA_VISIBLE_DEVICES=0 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 0.2
65.97   73.18   61.50   CUDA_VISIBLE_DEVICES=1 python -u train.coreference.py --task_name rte --do_train --do_lower_case --num_train_epochs 10 --train_batch_size 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 250 --seed 42 --kshot 0.1

90.78, 90.21, 89.1, 87.85, 86.08, 86.17, 83.4, 83.76, 74.39, 66.88

'''
