Doc2Query + PyTerrier
=========================================

`pyterrier-doc2query <https://github.com/terrierteam/pyterrier_doc2query>`__ provides PyTerrier
transformers for Doc2Query and related methods.

.. code-block:: console
	:caption: Install ``pyterrier-doc2query`` with pip

	pip install pyterrier-doc2query

What does it do?
----------------------------------------

A :class:`~pyterrier_doc2query.Doc2Query` transformer takes the text of each document and generates questions for that text.

.. code-block:: python

	import pyterrier_doc2query
	doc2query = pyterrier_doc2query.Doc2Query()
    sample_doc = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated"
    doc2query([{"docno" : "d1", "text" : sample_doc}])

The resulting dataframe will have an additional ``"querygen"`` column, which contains the generated queries, such as:

.. list-table::
   :header-rows: 1

   * - docno
     - querygen
   * - "d1"
     - 'what was the importance of the manhattan project to the united states atom project? what influenced the success of the united states why was the manhattan project a success? why was it important'

As a PyTerrier transformer, there are many ways to introduce Doc2Query into a PyTerrier retrieval process.

By default, the plugin loads `macavaney/doc2query-t5-base-msmarco <https://huggingface.co/macavaney/doc2query-t5-base-msmarco>`_, which is a version of `the checkpoint released by the original authors <https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-base.zip>`_ converted to PyTorch format. You can load another T5 model by passing another HuggingFace model name (or path to a model on the file system) as the first argument:

.. code-block:: python

    doc2query = pyterrier_doc2query.Doc2Query('some/other/model')

Using Doc2Query for Indexing
----------------------------------------

Indexing is as easy as instantiating the ``Doc2Query`` object and a PyTerrier indexer:

.. code-block:: python

    import pyterrier as pt
    dataset = pt.get_dataset("irds:vaswani")
    import pyterrier_doc2query
    doc2query = pyterrier_doc2query.Doc2Query(append=True) # append generated queries to the original document text
    indexer = doc2query >> pt.IterDictIndexer(index_loc)
    indexer.index(dataset.get_corpus_iter())

.. info::

	The generation process is expensive. Consider using :class:`pyterrier_caching.IndexerCache` to cache the
	generated queries in case you need them again.

Doc2Query--: When Less is More
----------------------------------------

The performance of Doc2Query can be significantly improved by removing queries that are not relevant to the documents that generated them. This involves first scoring the generated queries (using ``QueryScorer``) and then filtering out the least relevant ones (using ``QueryFilter``).

.. code-block:: python

    from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
    from pyterrier_dr import ElectraScorer

    doc2query = Doc2Query(append=False, num_samples=5)
    scorer = ElectraScorer()
    indexer = pt.IterDictIndexer('./index')
    pipeline = doc2query >> QueryScorer(scorer) >> QueryFilter(t=3.21484375) >> indexer # t=3.21484375 is the 70th percentile for generated queries on MS MARCO

    pipeline.index(dataset.get_corpus_iter())

We've also released pre-computed filter scores for various models on HuggingFace datasets:

- `macavaney/d2q-msmarco-passage-scores-electra <https://huggingface.co/datasets/macavaney/d2q-msmarco-passage-scores-electra>`_
- `macavaney/d2q-msmarco-passage-scores-monot5 <https://huggingface.co/datasets/macavaney/d2q-msmarco-passage-scores-monot5>`_
- `macavaney/d2q-msmarco-passage-scores-tct <https://huggingface.co/datasets/macavaney/d2q-msmarco-passage-scores-tct>`_

Using Doc2Query for Retrieval
----------------------------------------

Doc2Query can also be used at retrieval time (i.e., on retrieved documents) rather than at indexing time.

.. code-block:: python

    import pyterrier_doc2query
    doc2query = pyterrier_doc2query.Doc2Query()

    dataset = pt.get_dataset("irds:vaswani")
    bm25 = pt.terrier.Retriever.from_dataset("vaswani", "terrier_stemmed", wmodel="BM25")
    bm25 >> pt.get_text(dataset) >> doc2query >> pt.text.scorer(body_attr="querygen", wmodel="BM25")


References
----------------------------------------

.. cite:: doct5query
	:citation: Nogueira and Lin. From doc2query to docTTTTTquery. 2019.
	:link: https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf

.. cite.dblp:: conf/ecir/GospodinovMM23
