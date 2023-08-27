from __future__ import annotations
import os
import ast
from preprocessor import *
from index_db import *
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh = logging.FileHandler('../logs/retriever_benchmark.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

class Retriever(Preprocessor, IndexDB):
    def __init__(self, dir_path, embedding_model, tokenizer_model,
                 max_token_length, strides, embedding_path,
                 top_k, d, m, nbits) -> None:
        self.embedding_path = embedding_path
        self.topK = top_k
        self.pp = Preprocessor(dir_path, embedding_model,
                               tokenizer_model, max_token_length,
                               strides)
        self.index = IndexDB(d, m, nbits)

    def update_topK(selfself, top_k: int) -> None:
        self.topK = top_k

    def update_indexDB(self, embeddings: np.ndarray | str, is_read_from_file: bool=False) -> None:
        if is_read_from_file:
            embeds = pd.read_csv(embeddings)
            embeds['embedding'] = embeds.embedding \
                                        .apply(lambda x: x[0] + ','.join(x[1:-1].split()) + x[-1]) \
                                        .apply(lambda x: ast.literal_eval(x))
            embeddings = np.array(embeds.embedding.values.tolist())
        self.index.train(embeddings)
        self.index.add(embeddings)

    def get_lookup_df(self, embeddings: np.ndarray, filenames: List[str], chunks: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({
            'idx': range(embeddings.shape[0]),
            'filename': filenames,
            'original_text': chunks,
        })
        return df

    def get_user_embeddings(self, user_query: str) -> np.ndarray:
        uq_embs = self.pp.get_embeddings([user_query])
        return uq_embs

    def get_context_from_retriever(self, uq_embs, df: pd.DataFrame=None, threshold: float=0.0) -> tuple[str, str]:
        if df is None:
            df = pd.read_csv(self.embedding_path)
        combos = self.index.search(uq_embs, self.topK)
        combos = [(score, idx) for score, idx in zip(combos[0][0].tolist(), combos[1][0].tolist())]
        combos = [(score, int(idx)) if score >= threshold else (0.0, int(idx)) for score, idx in combos]
        indices = [idx for score, idx in combos if score != 0]
        if self.topK == 1:
            reference = ' '.join(df.iloc[[i for i in indices], :].loc[:, 'filename'].values.tolist()).split('/')[-1]
            context = ' '.join(df.iloc[[i for i in indices],:].loc[:, 'original_text'].values.tolist())
        else:
            reference = df.iloc[[i for i in indices], :].loc[:, 'filename'].values.tolist()
            reference = [filename.split('/')[-1] for filename in reference]
            context = df.iloc[[i for i in indices],:].loc[:, 'original_text'].values.tolist()
        return context, reference


if __name__ == '__main__':
    dir_path = '../../notebooks/vector-search/data/kb_extract'
    embedding_model = '../../notebooks/vector-search/all-mpnet-base-v2'
    tokenizer_model = '../../notebooks/vector-search/all-mpnet-base-v2'
    strides = 50
    logging.basicConfig(level=logging.INFO, force=True)
    user_query = 'I am getting error message in AWS'
    embedding_path = '../data/embedding_lc.csv'
    top_k = 5
    d = EMBEDDING_DIMENSION
    m = 8
    nbits = 8
    for max_token_length in [200, 300, 380]:
        r1 = Retriever(dir_path, embedding_model,
                       tokenizer_model, max_token_length,
                       strides, embedding_path, top_k,
                       d, m, nbits)
        filenames, texts = r1.pp.scan_kb_by_titles()
        filenames, chunks = r1.pp.get_tokenized_texts(filenames, texts, True)
        e1 = r1.pp.get_embedding(chunks)
        r1.update_indexDB(e1)
        e1_df = r1.get_lookup_df(e1, filenames, chunks)
        uq_e1 = r1.get_user_embeddings(user_query)
        embeddings_df = r1.get_lookup_df(e1, filenames, chunks)
        test_cases = [...]
        answers = [...]
        for i, (user_query, ans) in enumerate(zip(test_cases, answers), 1):
            uq_embeds = r1.get_user_embeddings(user_query)
            print(set(r1.get_context_from_retriever(uq_embeds, embeddings_df)[-1]))
