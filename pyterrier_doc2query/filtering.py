import numpy as np
import pandas as pd
import json
import ir_datasets
import math
import pyterrier as pt
import torch
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from more_itertools import chunked
from typing import List
import re
import more_itertools
import itertools
import torch
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config


class QueryScorer(pt.Transformer):
    def __init__(self, scorer):
        self.scorer = scorer

    def transform(self, inp):
          slices = []
          scorer_inp = {
              'query': [],
              'text': [],
          }
          for text, querygen in zip(inp['text'], inp['querygen']):
              queries = querygen.split('\n')
              start_idx = len(scorer_inp['query'])
              slices.append(slice(start_idx, start_idx+len(queries)))
              scorer_inp['query'].extend(queries)
              scorer_inp['text'].extend([text] * len(queries))
          scorer_inp['qid'] = list(range(len(scorer_inp['query'])))
          dout = self.scorer(pd.DataFrame(scorer_inp))
          return inp.assign(querygen_score=[dout['score'].values[s] for s in slices])


class QueryFilter(pt.Transformer):
    def __init__(self, t, append=True):
        self.t = t
        self.append = append

    def transform(self, inp):
        assert all(c in inp.columns for c in ['querygen', 'querygen_score'])
        inp = inp.reset_index(drop=True)
        querygen = ['\n'.join(np.array(qs.split('\n'))[ss >= self.t].tolist()) for qs, ss in zip(inp['querygen'], inp['querygen_score'])]
        if self.append:
            inp = inp.assign(text=inp['text'] + '\n' + pd.Series(querygen))
            inp = inp.drop(['querygen', 'querygen_score'], axis='columns')
        else:
            querygen_score = inp['querygen_score'].apply(lambda row: row[row >= t])
            inp = inp.assign(querygen=querygen, querygen_score=querygen_score)
        return inp
