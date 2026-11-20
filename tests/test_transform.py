import sys
import types
import unittest
from unittest import mock

import pandas as pd
import pyterrier_doc2query
import pyterrier_doc2query.itqg as itqg_module
import pyterrier_alpha as pta

class TestTransform(unittest.TestCase):
    def test_transform(self):
        doc2query = pyterrier_doc2query.Doc2Query()
        input = pd.DataFrame([
            {'docno': '1', 'text': 'Hello Terrier!'}
        ])
        out = doc2query(input)
        self.assertTrue('querygen' in out.columns)
        self.assertTrue('terrier' in out['querygen'][0].lower())

        # append mode
        doc2query.append = True
        out = doc2query(input)
        self.assertFalse('querygen' in out.columns)
        self.assertTrue(out['text'][0].startswith('Hello Terrier!'))

    def test_empty_input(self):
        doc2query = pyterrier_doc2query.Doc2Query()
        res = doc2query(pd.DataFrame([], columns=['docno', 'text', 'something_else']))
        self.assertEqual(list(res.columns), ['docno', 'text', 'something_else', 'querygen'])
        doc2query.append = True
        res = doc2query(pd.DataFrame([], columns=['docno', 'text', 'something_else']))
        self.assertEqual(list(res.columns), ['docno', 'text', 'something_else'])
        with self.assertRaises(pta.validate.InputValidationError):
            doc2query(pd.DataFrame([], columns=['docno', 'something_else'])) # missing text column


# the following fakes allow ITQG to be tested without vllm installed, or a GPU available

class _FakeSequence:
    def __init__(self, text):
        self.text = text


class _FakeLLMOutput:
    def __init__(self, texts):
        self.outputs = [_FakeSequence(t) for t in texts]


class _FakeLLM:
    instances = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.prompts = None
        self.sampling_params = None
        self.generate_calls = 0
        _FakeLLM.instances.append(self)

    def generate(self, prompts, sampling_params, **kwargs):
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.generate_kwargs = kwargs
        self.generate_calls += 1
        return [_FakeLLMOutput([f'{prompt.replace(chr(10), " ")} | query {j}' for j in range(sampling_params.n)]) for prompt in prompts]


class _FakeSamplingParams:
    def __init__(self, n, temperature, max_tokens):
        self.n = n
        self.temperature = temperature
        self.max_tokens = max_tokens


def _fake_vllm_module():
    mod = types.ModuleType('vllm')
    mod.LLM = _FakeLLM
    mod.SamplingParams = _FakeSamplingParams
    return mod


class _FakeTokenizer:
    """encodes each whitespace-separated word into two 'tokens'"""
    def __init__(self):
        self.template_calls = []
        self._word2id = {}

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
        ids = []
        for word in text.split():
            if word not in self._word2id:
                self._word2id[word] = len(self._word2id)
            ids.extend([2 * self._word2id[word], 2 * self._word2id[word] + 1])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens
        id2word = {v: k for k, v in self._word2id.items()}
        return ' '.join(id2word[i // 2] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize and add_generation_prompt
        self.template_calls.append(messages)
        return f'<|chat|>{messages[0]["content"]}'


class TestITQG(unittest.TestCase):
    _ABSENT = object()

    def setUp(self):
        _FakeLLM.instances = []
        self.tokenizer = _FakeTokenizer()
        patcher = mock.patch.object(itqg_module, 'AutoTokenizer')
        auto = patcher.start()
        auto.from_pretrained.return_value = self.tokenizer
        self.addCleanup(patcher.stop)
        self._saved_vllm = sys.modules.get('vllm', self._ABSENT)
        sys.modules['vllm'] = _fake_vllm_module()
        self.addCleanup(self._restore_vllm)

    def _restore_vllm(self):
        if self._saved_vllm is self._ABSENT:
            sys.modules.pop('vllm', None)
        else:
            sys.modules['vllm'] = self._saved_vllm

    def test_transform(self):
        itqg = pyterrier_doc2query.ITQG()
        self.assertEqual(itqg.checkpoint, 'luxinyayaya/ZeroGR-Query-Generator')
        input = pd.DataFrame([
            {'docno': '1', 'text': ' '.join(f't{i}' for i in range(10))},
            {'docno': '2', 'text': 'second document text'},
        ])
        out = itqg(input)
        self.assertTrue('querygen' in out.columns)
        q1 = out['querygen'][0].split('\n')
        q2 = out['querygen'][1].split('\n')
        self.assertEqual(len(q1), 16) # num_samples default
        self.assertEqual(len(q2), 16)
        for line in q1: # generated queries for the first document mention the first document
            self.assertIn('t0', line)
        for line in q2:
            self.assertIn('second', line)
            self.assertIn('text', line)
        llm = _FakeLLM.instances[0]
        self.assertEqual(llm.model, 'luxinyayaya/ZeroGR-Query-Generator') # default checkpoint
        self.assertEqual(llm.kwargs['dtype'], 'bfloat16')
        self.assertEqual(llm.kwargs['gpu_memory_utilization'], 0.5)
        self.assertEqual(llm.kwargs['max_model_len'], 1024)
        self.assertEqual(llm.sampling_params.n, 16)
        self.assertEqual(llm.sampling_params.temperature, 1.0)
        self.assertEqual(llm.sampling_params.max_tokens, 64)
        self.assertEqual(llm.generate_kwargs.get('use_tqdm'), False)
        prompt = llm.prompts[0]
        self.assertTrue(prompt.startswith('<|chat|>')) # prompt rendered with the chat template
        self.assertIn('You are a document to query generator.', prompt)
        self.assertIn('Retrieve document relevant to the given query.', prompt) # default instruction
        self.assertIn('t0', prompt) # document text in prompt
        self.assertEqual(self.tokenizer.template_calls[0], [{"role": "user", "content": prompt[len('<|chat|>'):]}])

    def test_append(self):
        itqg = pyterrier_doc2query.ITQG(append=True)
        input = pd.DataFrame([{'docno': '1', 'text': 'Hello Terrier!'}])
        out = itqg(input)
        self.assertFalse('querygen' in out.columns)
        self.assertTrue(out['text'][0].startswith('Hello Terrier!'))
        self.assertIn('query 0', out['text'][0])

    def test_empty_input(self):
        itqg = pyterrier_doc2query.ITQG()
        res = itqg(pd.DataFrame([], columns=['docno', 'text', 'something_else']))
        self.assertEqual(list(res.columns), ['docno', 'text', 'something_else', 'querygen'])
        self.assertEqual(_FakeLLM.instances[0].generate_calls, 0) # no generation performed
        itqg.append = True
        res = itqg(pd.DataFrame([], columns=['docno', 'text', 'something_else']))
        self.assertEqual(list(res.columns), ['docno', 'text', 'something_else'])
        with self.assertRaises(pta.validate.InputValidationError):
            itqg(pd.DataFrame([], columns=['docno', 'something_else'])) # missing text column

    def test_doc_truncation(self):
        itqg = pyterrier_doc2query.ITQG(max_doc_words=512) # relax the word cap, so the token cap applies
        doc = ' '.join(f'x{i}' for i in range(200)) # 200 words -> 400 tokens in the fake tokenizer
        itqg(pd.DataFrame([{'docno': '1', 'text': doc}]))
        prompt = _FakeLLM.instances[0].prompts[0]
        self.assertIn('x0', prompt)
        self.assertIn('x149', prompt) # 300 tokens // 2 tokens per word
        self.assertNotIn('x150', prompt)

    def test_no_vllm(self):
        sys.modules['vllm'] = None # makes `import vllm` raise ImportError
        with self.assertRaises(ImportError):
            pyterrier_doc2query.ITQG()
