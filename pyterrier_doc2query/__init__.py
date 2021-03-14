import pyterrier as pt
import torch
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from pyterrier.transformer import ApplyGenericTransformer
from more_itertools import chunked
from typing import List
import re
class Doc2Query(ApplyGenericTransformer):    
    def __init__(self, 
                 checkpoint='model.ckpt-1004000', 
                 base_model='t5-base', 
                 num_samples=3, 
                 batch_size=4, 
                 doc_attr="text", 
                 out_attr="querygen", 
                 verbose=True):
      
        self.num_samples = num_samples
        self.doc_attr = doc_attr
        self.out_attr = out_attr
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pattern = re.compile("^\\s*http\\S+")
        self.tokenizer = T5Tokenizer.from_pretrained(base_model)
        config = T5Config.from_pretrained(base_model)
        self.model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, from_tf=True, config=config)
        self.model.to(self.device)
        self.model.eval()

        def _add_attr(df):
          iter = chunked(df.itertuples(), self.batch_size)
          if self.verbose:
            iter = pt.tqdm(iter, total=len(df)/self.batch_size, unit='d')
          output=[]
          for batch_rows in iter:
            docs = [getattr(row, self.doc_attr) for row in batch_rows]
            gens = self._doc2query(docs)
            output.extend(gens)
          df[self.out_attr] = output
          return df
        super().__init__(_add_attr)

        print("Doc2query using %s" % str(self.device))

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
      rtr = []
      for i in range(0, len(docs)):
        offset = i * self.num_samples
        rtr.append(' '.join([self.tokenizer.decode(outputs[offset+j], skip_special_tokens=True) for j in range(self.num_samples)]))

      return rtr
