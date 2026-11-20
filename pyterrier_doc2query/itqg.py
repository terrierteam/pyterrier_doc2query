import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
from transformers import AutoTokenizer


class ITQG(pt.Transformer):
    """A :class:`~pyterrier.Transformer` that generates queries from documents using the instruction-tuned
    query generator (ITQG) of the ZeroGR framework for zero-shot generative retrieval (Sun et al., 2025).

    By default, this transformer uses the
    [`luxinyayaya/ZeroGR-Query-Generator`](https://huggingface.co/luxinyayaya/ZeroGR-Query-Generator)
    checkpoint, an instruction-tuned Llama-3.2-1B-Instruct trained on the OpenInstIR instruction corpus.
    Each document is presented to the model together with a natural-language task instruction, and the
    model generates a number of pseudo-queries per document.

    Inference is performed with [vLLM](https://docs.vllm.ai/), so a CUDA-compatible GPU is required.
    vLLM is an optional dependency of this package, and can be installed with:

        pip install pyterrier-doc2query[vllm]

    .. seealso::

        The inference procedure is adapted from
        `qg_vllm.py <https://github.com/luxinyayaya/ZeroGR/blob/fix/openinstir-checkpoints-and-e2e/qg_vllm.py>`_
        in the ZeroGR repository.
    """
    def __init__(self,
                 checkpoint='luxinyayaya/ZeroGR-Query-Generator',
                 num_samples=16,
                 doc_attr="text",
                 append=False,
                 out_attr="querygen",
                 verbose=False,
                 instruction='Retrieve document relevant to the given query.',
                 max_doc_words=128,
                 max_doc_tokens=300,
                 max_tokens=64,
                 temperature=1.0,
                 gpu_memory_utilization=0.5,
                 max_model_len=1024,
                 dtype='bfloat16',
                 **llm_kwargs):
        """
        Args:
            checkpoint: The checkpoint to use for the model. Defaults to 'luxinyayaya/ZeroGR-Query-Generator'.
            num_samples: The number of queries to generate per document.
            doc_attr: The attribute in the input DataFrame that contains the documents.
            append: If True, the generated queries are appended to the documents. Otherwise, the queries are stored in a separate attribute.
            out_attr: The attribute in the output DataFrame to store the generated queries.
            verbose: If True, displays a progress bar.
            instruction: The natural-language retrieval task instruction that conditions the query generation.
            max_doc_words: The maximum number of words of each document to use for query generation.
            max_doc_tokens: The maximum number of document tokens to use for query generation.
            max_tokens: The maximum number of tokens to generate per query.
            temperature: The temperature used when sampling the queries.
            gpu_memory_utilization: The GPU memory utilization, passed to vLLM's LLM.
            max_model_len: The maximum sequence length (context plus generated queries), passed to vLLM's LLM.
            dtype: The data type, passed to vLLM's LLM.
            llm_kwargs: Any additional keyword arguments passed to vLLM's LLM constructor.
        """
        self.checkpoint = checkpoint
        self.num_samples = num_samples
        self.doc_attr = doc_attr
        self.append = append
        self.out_attr = out_attr
        if append:
            assert out_attr == 'querygen', "append=True cannot be used with out_attr"
        self.verbose = verbose
        self.instruction = instruction
        self.max_doc_words = max_doc_words
        self.max_doc_tokens = max_doc_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("ITQG requires vLLM, which is not installed. Install it with `pip install vllm`, or `pip install pyterrier-doc2query[vllm]`.") from e
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.llm = LLM(model=checkpoint,
                       dtype=dtype,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len,
                       **llm_kwargs)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applied the query generation transformation."""
        with pta.validate.any(df) as v:
            v.document_frame(extra_columns=[self.doc_attr])
            v.result_frame(extra_columns=[self.doc_attr])
            v.columns(includes=[self.doc_attr])
        if len(df) == 0:
            output = []
        else:
            docs = list(df[self.doc_attr])
            output = self._itqg(docs)
            if self.append:
                output = [f'{doc}\n{gen}' for doc, gen in zip(docs, output)]
        if self.append:
            df = df.assign(**{self.doc_attr: output}) # replace doc content
        else:
            df = df.assign(**{self.out_attr: output}) # add new column
        return df

    def _itqg(self, docs):
        from vllm import SamplingParams
        prompts = [self._build_prompt(doc) for doc in docs]
        sampling_params = SamplingParams(n=self.num_samples,
                                         temperature=self.temperature,
                                         max_tokens=self.max_tokens)
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=self.verbose)
        return ['\n'.join(out.text.strip() for out in output.outputs) for output in outputs]

    def _build_prompt(self, doc):
        doc_text = " ".join(doc.split()[:self.max_doc_words])
        doc_text = self.tokenizer.decode(self.tokenizer.encode(doc_text, add_special_tokens=False)[:self.max_doc_tokens],
                                         skip_special_tokens=True)
        prompt = f"You are a document to query generator. For the retrieval task: {self.instruction}\n\nGenerate relevant search query for the following document:\n\n{doc_text}"
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False,
                                                  add_generation_prompt=True)
