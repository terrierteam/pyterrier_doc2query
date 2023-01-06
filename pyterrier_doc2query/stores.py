import contextlib
from pathlib import Path
import json
import numpy as np
import lz4.block
import pyterrier as pt
from npids import Lookup
from . import Artefact


class Doc2QueryStore(pt.Indexer, Artefact):
    def __init__(self, path):
        self.path = Path(path)
        self._meta = None
        self._queries = None
        self._queries_offsets = None
        self._docnos = None

    def payload(self):
        if self._meta is None:
            assert (self.path/'pt_meta.json').exists()
            with  (self.path/'pt_meta.json').open('rt') as fin:
                self._meta = json.load(fin)
            assert self._meta.get('type') == 'doc2query_store'
        if self._queries is None:
            self._queries = (self.path/'queries.lz4').open('rb')
        if self._queries_offsets is None:
            self._queries_offsets = np.memmap(self.path/'queries.offsets.u8', mode='r', dtype=np.uint64)
        if self._docnos is None:
            self._docnos = Lookup(self.path/'docnos.npids')
        return self._queries, self._queries_offsets, self._docnos

    def generator(self, limit_k=None, append=False):
        return Doc2QueryStoreGenerator(self, limit_k, append=append)

    def transform(self, inp):
        return self.generator()(inp)

    def index(self, inp_it):
        assert not (self.path/'pt_meta.json').exists()
        self.path.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            f_queries = stack.enter_context((self.path/'queries.lz4').open('wb'))
            f_queries_offsets = stack.enter_context((self.path/'queries.offsets.u8').open('wb'))
            f_docnos = stack.enter_context(Lookup.builder(self.path/'docnos.npids'))
            f_queries_offsets.write(np.array([f_queries.tell()], dtype=np.uint64).tobytes())
            for record in inp_it:
                f_queries.write(lz4.block.compress(record['querygen'].encode()))
                f_queries_offsets.write(np.array([f_queries.tell()], dtype=np.uint64).tobytes())
                f_docnos.add(record['docno'])
            with (self.path/'pt_meta.json').open('wt') as f_meta:
                json.dump({'type': 'doc2query_store'}, f_meta)
        return self

    def lookup(self, docnos, limit_k=None):
        single = False
        if isinstance(docnos, str):
            docnos = np.array([docnos])
            single = True
        queries, q_offsets, docnos_lookup = self.payload()
        dids = docnos_lookup.inv[docnos]
        if (dids == -1).sum() > 0:
            raise ValueError(f"unknown docno(s) encountered: {docnos[dids == -1]}")
        querygen = [_lz4_read(queries, s, e, limit_k) for s, e in zip(q_offsets[dids], q_offsets[dids+1])]
        if single:
            return {'querygen': querygen[0]}
        return {'querygen': querygen}

    def __iter__(self):
        queries, q_offsets, docnos = self.payload()
        for i, docno in enumerate(docnos):
            yield {'docno': docno, 'querygen': _lz4_read(queries, q_offsets[i], q_offsets[i+1], limit_k=None)}


class QueryScoreStore(pt.Indexer, Artefact):
    def __init__(self, path):
        self.path = Path(path)
        self._meta = None
        self._scores = None
        self._scores_offsets = None
        self._queries = None
        self._queries_offsets = None
        self._docnos = None
        self._doc2query_store = None

    def payload(self):
        if self._meta is None:
            assert (self.path/'pt_meta.json').exists()
            with  (self.path/'pt_meta.json').open('rt') as fin:
                self._meta = json.load(fin)
            assert self._meta.get('type') == 'query_score_store'
        if self._scores is None:
            self._scores = np.memmap(self.path/'scores.f2', mode='r', dtype=np.float16)
        if self._scores_offsets is None:
            self._scores_offsets = np.memmap(self.path/'scores.offsets.u8', mode='r', dtype=np.uint64)
        if self._meta.get('doc2query_store_repo'):
            if self._doc2query_store is None:
                self._doc2query_store = Doc2QueryStore.from_repo(self._meta['doc2query_store_repo'])
            queries, queries_offsets, docnos = self._doc2query_store.payload()
            if self._queries is None:
                self._queries = queries
            if self._queries_offsets is None:
                self._queries_offsets = queries_offsets
            if self._docnos is None:
                self._docnos = docnos
        else:
            if self._queries is None:
                self._queries = (self.path/'queries.lz4').open('rb')
            if self._queries_offsets is None:
                self._queries_offsets = np.memmap(self.path/'queries.offsets.u8', mode='r', dtype=np.uint64)
            if self._docnos is None:
                self._docnos = Lookup(self.path/'docnos.npids')
        return self._scores, self._scores_offsets, self._queries, self._queries_offsets, self._docnos

    def close(self):
        self._meta = None
        self._scores = None
        self._scores_offsets = None
        self._queries = None
        self._queries_offsets = None
        self._docnos = None
        self._doc2query_store = None

    def query_scorer(self, limit_k=None):
        return QueryScoreStoreScorer(self, limit_k)

    def transform(self, inp):
        return self.query_scorer()(inp)

    def index(self, inp_it):
        assert not (self.path/'pt_meta.json').exists()
        self.path.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            f_scores = stack.enter_context((self.path/'scores.f2').open('wb'))
            f_scores_offsets = stack.enter_context((self.path/'scores.offsets.u8').open('wb'))
            f_queries = stack.enter_context((self.path/'queries.lz4').open('wb'))
            f_queries_offsets = stack.enter_context((self.path/'queries.offsets.u8').open('wb'))
            f_docnos = stack.enter_context(Lookup.builder(self.path/'docnos.npids'))
            scores_count = 0
            f_scores_offsets.write(np.array([scores_count], dtype=np.uint64).tobytes())
            f_queries_offsets.write(np.array([f_queries.tell()], dtype=np.uint64).tobytes())
            for record in inp_it:
                scores_count += len(record['querygen_score'])
                f_scores.write(record['querygen_score'].astype(np.float16).tobytes())
                f_scores_offsets.write(np.array([scores_count], dtype=np.uint64).tobytes())
                f_queries.write(lz4.block.compress(record['querygen'].encode()))
                f_queries_offsets.write(np.array([f_queries.tell()], dtype=np.uint64).tobytes())
                f_docnos.add(record['docno'])
            with (self.path/'pt_meta.json').open('wt') as f_meta:
                json.dump({'type': 'query_score_store'}, f_meta)
        return self

    def percentile(self, p):
        scores, s_offsets, queries, q_offsets, docnos = self.payload()
        return np.percentile(scores, p)

    def lookup(self, docnos, limit_k=None):
        single = False
        if isinstance(docnos, str):
            docnos = np.array([docnos])
            single = True
        scores, s_offsets, queries, q_offsets, docnos_lookup = self.payload()
        dids = docnos_lookup.inv[docnos]
        if (dids == -1).sum() > 0:
            raise ValueError(f"unknown docno(s) encountered: {docnos[dids == -1]}")
        querygen = [_lz4_read(queries, s, e, limit_k) for s, e in zip(q_offsets[dids], q_offsets[dids+1])]
        querygen_score = [np.array(scores[s:e][:limit_k]) for s, e in zip(s_offsets[dids], s_offsets[dids+1])]
        if single:
            return {'querygen': querygen[0], 'querygen_score': querygen_score[0]}
        return {'querygen': querygen, 'querygen_score': querygen_score}

    def __iter__(self):
        scores, s_offsets, queries, q_offsets, docnos = self.payload()
        for i, docno in enumerate(docnos):
            yield {'docno': docno, 'querygen': _lz4_read(queries, q_offsets[i], q_offsets[i+1], limit_k=None), 'querygen_score': np.array(scores[s_offsets[i]:s_offsets[i+1]])}


def _lz4_read(f, start, end, limit_k):
    if f.tell() != start:
        f.seek(start)
    buffer = f.read(end-start)
    queries = lz4.block.decompress(buffer).decode()
    if limit_k is not None:
        queries = '\n'.join(queries.split('\n')[:limit_k])
    return queries


class QueryScoreStoreScorer(pt.Transformer):
    def __init__(self, scorer_store, limit_k=None):
        self.store = scorer_store
        self.limit_k = limit_k

    def transform(self, inp):
        return inp.assign(**self.store.lookup(inp.docno, self.limit_k))


class Doc2QueryStoreGenerator(pt.Transformer):
    def __init__(self, d2q_store, limit_k=None, append=False):
        self.store = d2q_store
        self.limit_k = limit_k
        self.append = append

    def transform(self, inp):
        res = self.store.lookup(inp.docno, self.limit_k)
        if self.append:
            return inp.assign(text=inp['text'] + '\n' + res['querygen'])
        return inp.assign(**self.store.lookup(inp.docno, self.limit_k))
