__version__ = '0.2.0'

import math
import pyterrier as pt
import pyterrier_alpha as pta
import pandas as pd
import torch
from transformers import T5Tokenizer, T5TokenizerFast, T5ForConditionalGeneration
from more_itertools import chunked
from typing import List
import re
from warnings import warn
from .artefact import Artefact
from .filtering import QueryScorer, QueryFilter
from .stores import Doc2QueryStore, QueryScoreStore


class Doc2Query(pt.Transformer):
    """A :class:`~pyterrier.Transformer` that generates queries from documents."""
    def __init__(self,
                 checkpoint='macavaney/doc2query-t5-base-msmarco',
                 num_samples=3,
                 batch_size=4,
                 doc_attr="text",
                 append=False,
                 out_attr="querygen",
                 verbose=False,
                 fast_tokenizer=False,
                 device=None):
        """
        Args:
            checkpoint: The checkpoint to use for the model. Defaults to 'macavaney/doc2query-t5-base-msmarco'.
            num_samples: The number of queries to generate per document.
            batch_size: The batch size to use for inference.
            doc_attr: The attribute in the input DataFrame that contains the documents.
            append: If True, the generated queries are appended to the documents. Otherwise, the queries are stored in a separate attribute.
            out_attr: The attribute in the output DataFrame to store the generated queries.
            verbose: If True, displays a progress bar.
            fast_tokenizer: If True, uses the fast tokenizer.
            device: The device to use for inference. If None, defaults to 'cuda' if available, otherwise 'cpu'.
        """
        self.checkpoint = checkpoint
        self.num_samples = num_samples
        self.doc_attr = doc_attr
        self.append = append
        self.out_attr = out_attr
        if append:
          assert out_attr == 'querygen', "append=True cannot be used with out_attr"
        self.verbose = verbose
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.pattern = re.compile("^\\s*http\\S+")
        if fast_tokenizer:
            self.tokenizer = T5TokenizerFast.from_pretrained(checkpoint)
        else:
            warn('consider setting fast_tokenizer=True; it speeds up inference considerably')
            self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.fast_tokenizer = fast_tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applied the query generation transformation."""
        with pta.validate.any(df) as v:
            v.document_frame(extra_columns=[self.doc_attr])
            v.result_frame(extra_columns=[self.doc_attr])
            v.columns(includes=[self.doc_attr])
        it = chunked(iter(df[self.doc_attr]), self.batch_size)
        if self.verbose and len(df) > 0:
            it = pt.tqdm(it, total=math.ceil(len(df)/self.batch_size), unit='d', desc='doc2query')
        output = []
        for docs in it:
            docs = list(docs) # so we can refernece it again when self.append
            gens = self._doc2query(docs)
            if self.append:
                gens = [f'{doc}\n{gen}' for doc, gen in zip(docs, gens)]
            output.extend(gens)
        if self.append:
            df = df.assign(**{self.doc_attr: output}) # replace doc content
        else:
            df = df.assign(**{self.out_attr: output}) # add new column
        return df

    def _doc2query(self, docs : List[str]):
        docs = [re.sub(self.pattern, "", doc) for doc in docs]
        with torch.no_grad():
          input_ids = self.tokenizer(docs,
                                     max_length=64,
                                     return_tensors='pt',
                                     padding=True,
                                     truncation=True).input_ids.to(self.device)
          outputs = self.model.generate(
              input_ids=input_ids,
              max_length=64,
              do_sample=True,
              top_k=10,
              num_return_sequences=self.num_samples)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rtr = ['\n'.join(gens) for gens in chunked(outputs, self.num_samples)]
        return rtr

__all__ = ['Doc2Query', 'QueryScorer', 'QueryFilter', 'Doc2QueryStore', 'QueryScoreStore', 'Artefact']
