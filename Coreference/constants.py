# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


"""Constants.
"""

from enum import Enum


class Gender(Enum):
  UNKNOWN = 0
  MASCULINE = 1
  FEMININE = 2


# Mapping of (lowercased) pronoun form to gender value. Note that reflexives
# are not included in GAP, so do not appear here.
PRONOUNS = {
    'she': Gender.FEMININE,
    'her': Gender.FEMININE,
    'hers': Gender.FEMININE,
    'he': Gender.MASCULINE,
    'his': Gender.MASCULINE,
    'him': Gender.MASCULINE,
}

# Fieldnames used in the gold dataset .tsv file.
GOLD_FIELDNAMES = [
    'ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'A-coref', 'B',
    'B-offset', 'B-coref', 'URL'
]

# Fieldnames expected in system output .tsv files.
SYSTEM_FIELDNAMES = ['ID', 'A-coref', 'B-coref']
