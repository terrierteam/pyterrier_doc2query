# PyTerrier_doc2query

**New: Check out our interactive demo on [🤗 HuggingFace Spaces](https://huggingface.co/spaces/macavaney/pyterrier_doc2query)**

**New: Improve effectiveness and efficiency using Doc2Query&minus;&minus;**

This is the [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for the [docTTTTTquery](https://github.com/castorini/docTTTTTquery) [Nogueira20] and Doc2Query&minus;&minus; [Gospodinov23] approaches for document expansion by query prediction.

## Installation

This repostory can be installed using Pip.

    pip install --upgrade git+https://github.com/terrierteam/pyterrier_doc2query.git

## What does it do?

A Doc2Query object has a transform() function, which takes the text of each document, and suggests questions
for that text. 

```python

sample_doc = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated"

import pyterrier_doc2query
doc2query = pyterrier_doc2query.Doc2Query()
doc2query([{"docno" : "d1", "text" : sample_doc}])

```

The resulting dataframe will have an additional `"querygen"` column, which
contains the generated queries, such as:

| docno | querygen  |
|-------|-----------|
| "d1"  | 'what was the importance of the manhattan project to the united states atom project? what influenced the success of the united states why was the manhattan project a success? why was it important' |

As a PyTerrier transformer, there are lots of ways to introduce Doc2query into a PyTerrier retrieval
process.

By default, the plugin loads [`macavaney/doc2query-t5-base-msmarco`](https://huggingface.co/macavaney/doc2query-t5-base-msmarco), which is a a version of [the checkpoint released by the original authors](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-base.zip), converted to pytorch format.
You can load another T5 model by passing another huggingface model name (or path to model on the file system) by passing it as the first argument:

```python
doc2query = pyterrier_doc2query.Doc2Query('some/other/model')
```

## Using Doc2Query for Indexing


Then, indexing is as easy as instantiating the Doc2Query object and a PyTerrier indexer:

```python
import pyterrier as pt
dataset = pt.get_dataset("irds:vaswani")
import pyterrier_doc2query
doc2query = pyterrier_doc2query.Doc2Query(append=True) # append generated queries to the orignal document text
indexer = doc2query >> pt.IterDictIndexer(index_loc)
indexer.index(dataset.get_corpus_iter())
```

## Doc2Query&minus;&minus;: When Less is More

The performance of Doc2Query can be significantly improved by removing queries that are not relevant to the
documents that generated them. This involves first scoring the generated queries (using `QueryScorer`) and
then filtering out the least relevant ones (using `QueryFilter`).

```python
from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
from pyterrier_dr import ElectraScorer

doc2query = Doc2Query(append=False, num_samples=5)
scorer = ElectraScorer()
indexer = pt.IterDictIndexer('./index')
pipeline = doc2query >> QueryScorer(scorer) >> QueryFilter(t=3.21484375) >> indexer # t=3.21484375 is the 70th percentile for generated queries on MS MARCO

pipeline.index(dataset.get_corpus_iter())
```

We've also released pre-computed filter scores for various models on HuggingFace datasets:
 - [macavaney/d2q-msmarco-passage-scores-electra](https://huggingface.co/datasets/macavaney/d2q-msmarco-passage-scores-electra)
 - [macavaney/d2q-msmarco-passage-scores-monot5](https://huggingface.co/datasets/macavaney/d2q-msmarco-passage-scores-monot5)
 - [macavaney/d2q-msmarco-passage-scores-tct](https://huggingface.co/datasets/macavaney/d2q-msmarco-passage-scores-tct)

## Using Doc2Query for Retrieval

Doc2query can also be used at retrieval time (i.e. on retrieved documents) rather than 
at indexing time.

```python

import pyterrier_doc2query
doc2query = pyterrier_doc2query.Doc2Query()

dataset = pt.get_dataset("irds:vaswani")
bm25 = pt.terrier.Retriever.from_dataset("vaswani", "terrier_stemmed", wmodel="BM25")
bm25 >> pt.get_text(dataset) >> doc2query >> pt.text.scorer(body_attr="querygen", wmodel="BM25")

```

## ZeroGR Query Generator (ITQG)

In addition to Doc2Query, the plugin also includes the instruction-tuned query generator from
[ZeroGR](https://github.com/luxinyayaya/ZeroGR) (Sun et al. 2025), a framework for zero-shot generative
retrieval. It is based on a Llama-3.2-1B-Instruct checkpoint fine-tuned on the OpenInstIR instruction corpus,
and generates diverse pseudo-queries from a document and a natural-language task instruction. `ITQG` is used in the same way
as `Doc2Query`:

```python
import pyterrier_doc2query
itqg = pyterrier_doc2query.ITQG()
itqg([{"docno" : "d1", "text" : sample_doc}])
```

The resulting dataframe will have an additional `"querygen"` column (or, with `append=True`, the generated
queries will be appended to the document text).

By default, `ITQG` loads [`luxinyayaya/ZeroGR-Query-Generator`](https://huggingface.co/luxinyayaya/ZeroGR-Query-Generator).
You can pass a different checkpoint, number of queries per document, or a task instruction:

```python
itqg = pyterrier_doc2query.ITQG(checkpoint='some/other/qg', num_samples=8,
                                instruction='Retrieve document relevant to the given query.')
```

`ITQG` performs inference with [vLLM](https://docs.vllm.ai/), so a CUDA-compatible GPU is required. vLLM is
an optional dependency of the package, and is not installed by default:

    pip install pyterrier-doc2query[vllm]

## Examples

Check out out the notebooks, even on Colab:

 - Vaswani [[Github](https://github.com/terrierteam/pyterrier_doc2query/blob/master/pyterrier_doc2query_vaswani.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/pyterrier_doc2query/blob/master/pyterrier_doc2query_vaswani.ipynb)]

## Implementation Details

We use a PyTerrier transformer to rewrite documents by doc2query.

## References

  - [Nogueira20]: Rodrigo Nogueira and Jimmy Lin. From doc2query to docTTTTTquery. https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf
  - [Gospodinov23]: Mitko Gospodinov, Sean MacAvaney, and Craig Macdonald. Doc2Query--: When Less is More. ECIR 2023.
  - [Sun25]: Weiwei Sun, Keyi Kong, Xinyu Ma, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, Zhaochun Ren, Yiming Yang. ZeroGR: A Generalizable and Scalable Framework for Zero-Shot Generative Retrieval. arXiv preprint arXiv:2510.10419, 2025.
  - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation inInformation Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271

## Credits

- Craig Macdonald, University of Glasgow
- Sean MacAvaney, University of Glasgow
- Mitko Gospodinov, University of Glasgow
