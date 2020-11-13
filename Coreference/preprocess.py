# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import csv
import random
def load_GAP_coreference_data(filename, k_shot):
    path = '/export/home/Dataset/gap_coreference/'

    def generate_hypothesis(sentence, pronoun_str, pronoun_position, entity_str, entity_position):
        pronoun_len = len(pronoun_str)
        entity_len = len(entity_str)
        # print(sentence, type(pronoun_position), type(pronoun_len), type(pronoun_str))
        assert sentence[pronoun_position: (pronoun_position+pronoun_len)] == pronoun_str
        assert sentence[entity_position: (entity_position+entity_len)] == entity_str

        pronoun_left_context = sentence[:pronoun_position]
        pronoun_right_context = sentence[pronoun_position+pronoun_len:]
        # print('pronoun_left_context:', pronoun_left_context)
        # print('pronoun_right_context:', pronoun_right_context)
        if pronoun_str in set(['her','Her', 'his', 'His']):
            entity_str=entity_str+"'s"
        hypothesis = pronoun_left_context.strip()+' '+entity_str+' '+pronoun_right_context.strip()

        return hypothesis


    all_examples = []
    with open(path+filename) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            all_examples.append(row)

    '''select k examples'''
    if k_shot > 0:
        selected_examples = random.sample(all_examples, k_shot)
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
        hypy_A = generate_hypothesis(premise, pronoun, pronoun_pos, entity_A, entity_A_pos)

        entity_B = example['B']
        entity_B_pos = int(example['B-offset'])
        entity_B_label = example['B-coref']
        hypy_B = generate_hypothesis(premise, pronoun, pronoun_pos, entity_B, entity_B_pos)

        selected_example_list.append((idd, premise, hypy_A, entity_A_label, hypy_B, entity_B_label))

    return selected_example_list

if __name__ == "__main__":
    load_GAP_coreference_data(10)
