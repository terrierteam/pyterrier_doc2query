# PyTerrier_doc2query

This is the [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for the [docTTTTTquery](https://github.com/castorini/docTTTTTquery) approach for document expansion by query prediction [Nogueira20].

## Installation

This repostory can be installed using Pip.

    pip install --upgrade git+https://github.com/terrierteam/pyterrier_doc2query.git

You will also need a fine-tuned checkpoint (unzipped) for the T5 model from the [docTTTTTquery repository](https://github.com/castorini/docTTTTTquery#data-and-trained-models-ms-marco-passage-ranking-dataset).


## What does it do?

A Doc2Query object has a transform() function, which takes the text of each document, and suggests questions
for that text. 

```

sample_doc = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated"

import pyterrier_doc2query
import pandas as pd
doc2query = pyterrier_doc2query.Doc2Query("/path/to/checkpoint", out_attr="text")
doc2query.transform(pd.DataFrame([{"docno" : "d1", "text" : sample_doc]]))

```

The resulting dataframe returned by transform() will have an additional `"query_gen"` column, which
contains the generated queries, such as:

|-------|-----------|
| docno | querygen  |
|-------|-----------|
| "d1"  | 'what was the importance of the manhattan project to the united states atom project? what influenced the success of the united states why was the manhattan project a success? why was it important' |

As a PyTerrier transformer, there are lots of ways to introduce Doc2query into a PyTerrier retrieval
process.

## Using Doc2Query for Indexing


Then, indexing is as easy as instantiating the Doc2Query object and a PyTerrier indexer, pointing at the (unzipped) checkpoint and the directory in which you wish to create an index.

```python

dataset = pt.get_dataset("irds:vaswani")
import pyterrier_doc2query
doc2query = pyterrier_doc2query.Doc2Query("/path/to/checkpoint", out_attr="text")
indexer = doc2query >> pt.IterDictIndexer(index_loc)
indexer.index(dataset.get_corpus_iter())
```

## Using Doc2Query for Retrieval

Doc2query can also be used at retrieval time (i.e. on retrieved documents) rather than 
at indexing time.

```python

import pyterrier_doc2query
doc2query = pyterrier_doc2query.Doc2Query("/path/to/checkpoint", out_attr="querygen")


dataset = pt.get_dataset("irds:vaswani")
bm25 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
bm25 >> pt.get_text(dataset) >> doc2query >> pt.text.scorer(body_attr="querygen", wmodel="BM25")

```


## Examples

Checkout out the notebooks, even on Colab:

 - TODO

## Implementation Details

We use a PyTerrier transformer to rewrite documents by doc2query.

## References

  - [Nogueira20]: Rodrigo Nogueira and Jimmy Lin. From doc2query to docTTTTTquery. https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf
  - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation inInformation Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271

## Credits

- Craig Macdonald, University of Glasgow
