from __future__ import annotations
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import sys


FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level="INFO")
logger = logging.getLogger("global_logger")

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import models, SentenceTransformer
from tqdm.auto import tqdm
from reader import *


class Metrics:
    def __init__(self, retriever_model: str=None, reader_model: str=None) -> None:
        self.retriever_model = SentenceTransformer(retriever_model)
        self.reader_model = Predictor(
            endpoint_name=LLM_ENDPOINT,
            serializer=JSONSerializer,
            deserializer=JSONDeserializer,
        )

    def get_eval_dataset(self, data_path: str) -> 'datasets.arrow_dataset.Dataset':
        custom_datasets = load_dataset('csv', data_files=data_path)
        custom = custom_datasets['train'].train_test_split(test_size=0.9, seed=123)
        custom_eval = custom['test']
        return custom_eval

    def get_eval_df(self, custom_eval: 'datasets.arrow_dataset.Dataset') -> pd.DataFrame:
        for i, row in tqdm(enumerate(custom_eval)):
            if i == 0:
                custom_df = pd.DataFrame([row])
            else:
                custom_df = pd.concat([custom_df, pd.DataFrame([row])],
                                      ignore_index=True)
        # custom_df = pd.DataFrame()
        # for i, row in tqdm(enumerate(custom_eval)):
        #     if i == 0:
        #         custom_df = pd.DataFrame(data=np.array([[row['question'],row['context'],row['id']]]),
        #                                  columns=['question','context','id'])
        #     else:
        #         custom_df = pd.concat([pd.DataFrame(data=np.array([[row['question'],row['context'],row['id']]]),
        #                                             columns['question','context','id']),
        #                                custom_df], ignore_index=True)
        no_dupe = custom_df.drop_duplicates(
            subset='context',
            keep='first',
        )
        # also drop question column
        no_dupe = no_dupe.drop(columns=['question'])
        # and give each context a slightly unique ID
        try:
            no_dupe['id'] = no_dupe['id'] + 'con'
        except:
            pass
        custom_df = custom_df.merge(no_dupe, how='inner', on='context')
        custom_df.head()
        return custom_df

    def prepare_eval_mappings(self, custom_df: pd.DataFrame):
        ir_queries = {
            row['id_x']: row['question'] for i, row in custom_df.iterrows()
        }
        ir_corpus = {
            row['id_y']: row['context'] for i, row in custom_df.iterrows()
        }
        ir_relevant_docs = {key: [] for key in custom_df['id_x'].unique()}
        for i, row in custom_df.iterrows():
            # we append in the case of a question ID being connected to
            # multiple context IDs
            ir_relevant_docs[row['id_x']].append(row['id_y'])
        # this must be in format {question_id: {set of context_ids}}
        ir_relevant_docs = {key: set(values) for key, values in ir_relevant_docs.items()}
        return ir_queries, ir_corpus, ir_relevant_docs

    def get_ir_eval(self, ir_queries, ir_corpus, ir_relevant_docs, dataset_name: str='TEST'):
        ir_eval = InformationRetrievalEvaluator(
            ir_queries, ir_corpus, ir_relevant_docs, name=dataset_name, write_csv='abc.csv'
        )
        return ir_eval

    def compute_ir(self, ir_eval, write_path: str=None):
        return ir_eval(self.retriever_model, write_path)

    def get_cosim_scores(self, contexts1: str | List[str], contexts2: str | List[str],
                         threshold: float=0.0) -> List[float]:
        if isinstance(contexts1, str):
            contexts1 = [contexts1]
        if isinstance(contexts2, str):
            contexts2 = [contexts2]
        n, m = len(contexts1), len(contexts2)
        if n < m:
            contexts1 = contexts1 + (m-n) * contexts1
        embed1 = self.retriever_model.encode(contexts1)
        embed2 = self.retriever_model.encode(contexts2)
        labels = []
        cosine_scores = 1 - (paired_cosine_distances(embed1, embed2))
        return [score if score >= threshold else 0.0 for score in cosine_scores]

    def get_embedding_cosim(self, eval_df):
        return EmbeddingSimilarityEvaluator(eval_df.question.values.tolist(),
                                            eval_df.question.values.tolist(),
                                            [0.1 for _ in range(eval_df.shape[0])])


if __name__ == '__main__':
    retriever_model = 'all-mpnet-base-v2'
    data_path = 'question_generator/kb_qa_gen2/data_0.csv'
    m = Metrics(retriever_model=retriever_model)
    custom_eval = m.get_eval_dataset(data_path)
    custom_df = m.get_eval_df(custom_eval)
    custom_df['texts'] = custom_df['question']
    ir_queries, ir_corpus, ir_relevant_docs = m.prepare_eval_mappings(custom_df)
    ir_eval = m.get_ir_eval(ir_queries, ir_corpus, ir_relevant_docs, 'pre-trained TEST')
    ir_score = m.compute_ir(ir_eval, '../data')
