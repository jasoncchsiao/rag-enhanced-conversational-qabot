#!/usr/bin/python

from __future__ import annotations
import os
import ast
from preprocessor import *
from index_db import *
from retriever import *

import logging
import sys
from datetime import datetime
import pickle
import json

t0 = datetime.now()
questions = [...]
answers = []

dir_path = '../data/source'
tokenizer_model = 'sentence-transformers/all-mpnet-base-v2'
strides = 50
logging.basicConfig(level=logging.INFO, force=True)
embedding_path = '../data/saved_embeddings.csv'
top_k = 5
d = EMBEDDING_DIMENSION
m = 32
nbits = 8
hashmaps = []
embedding_model = 'sentence-transformers/all-mpnet-base-v2'
max_token_length = 300

with open('models/retrievers/fine-tuned/quant-32.pkl', 'rb') as f:
    r1 = pickle.load(f)

filenames, texts = r1.pp.scan_kb_by_titles(False)
filenames, chunks = r1.pp.get_tokenized_texts(filenames, texts, False)
embeds = pp.read_csv(embedding_path)
embeds['embedding'] = embeds.embedding \
                            .apply(lambda x: x[0] + ','.join(x[1:-1].split()) + x[-1]) \
                            .apply(lambda x: ast.literal_eval(x))
e1 = np.array(embeds.embedding.values.tolist())
embeddings_df = r1.get_lookup_df(e1, filenames, chunks)

for i, (user_query, ans) in enumerate(zip(test_cases, answers), 1):
    uq_embeds = r1.get_user_embeddings(user_query)
    chunks, rels = r1.get_context_from_retriever(uq_embeds, embeddings_df, 3)

hashmap = {}
top_ks = [1, 5, 10, 15, 30, 50]
for top_k in top_ks:
    count = 0
    r1.update_topK(top_k)
    for i, (user_query, ans) in enumerate(zip(test_cases, answers), 1):
        uq_embeds = r1.get_user_embeddings(user_query)
        chunks, rels = r1.get_context_from_retriever(uq_embeds, embeddings_df)
        hashmap['chunks@{}@test_case{}'.format(str(top_k), str(i))] = (user_query, chunks)
        hashmap['refs@{}@test_case{}'.format(str(top_k), str(i))] = (user_qeury, rels)
        if ans in rels:
            count += 1
    hashmap['pass_rate_@{}'.format(str(top_k))] = count*1.0/len(answers)
hashmaps.append(hashmap)
with open('../data/optimal_retriever_result.csv', 'w') as f:
    json.dump(hashmaps, f)
print('total compute time'.upper(), datetime.now() - t0)
