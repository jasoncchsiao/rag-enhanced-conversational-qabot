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
with open('models/retrievers/fine-tuned/quant-32.pkl', 'rb') as f:
    r1 = pickle.load(f)
import pandas as pd
from datetime import datetime
import json
data_paths = ['../data/question-generator/qa_gen/data_0.csv']
t0 = datetime.now()
embedding_path = '../data/saved_embeddings.csv'
history = []
for j, data_path in enumerate(data_paths):
    data = pd.read_csv(data_path)
    answers = data.title.values.tolist()
    test_cases = data.question.values.tolist()
    thresholds = [i for i in range(9)][::-1]
    topKs = [i+1 for i in range(15)] # From small k to large k
    filenames, texts = r1.pp.scan_kb_by_titles(False)
    filenames, chunks = r1.pp.get_tokenized_texts(filenames, texts, True)
    embeds = pd.read_csv(embedding_path)
    embeds['embedding'] = embeds.embedding \
                                .apply(lambda x: x[0] + ','.join(x[1:-1].split()) + x[-1]) \
                                .apply(lambda x: ast.literal_eval(x))
    e1 = np.array(embeds.embedding.values.tolist())
    embeddings_df = r1.get_lookup_df(e1, filenames, chunks)

    best_thresh = None
    best_topK = None
    best_pass_rate = 0.0
    for thresh in thresholds:
        for topK in topKs:
            r1.update_topK(topK)
            count = 0
            for i, (user_query, ans) in enumerate(zip(test_cases, answers), 1):
                uq_embeds = r1.get_user_embeddings(user_query)
                chunks, rels = r1.get_context_from_retriever(uq_embeds, embeddings_df, thresh)
                if ans in rels:
                    count += 1
            pass_rate = count * 1.0 / len(answers)
            history.append(best_thresh, best_topK, best_pass_rate)
            if pass_rate > best_pass_rate:
                best_pass_rate = pass_rate
                best_thresh = thresh
                best_topK = topK
    hashmap = {}
    hashmap['thresholds'] = thresholds
    hashmap['topKs'] = topKs
    hashmap['best_threshold'] = best_thresh
    hashmap['best_topK'] = best_topK
    hashmap['best pass rate'] = best_pass_rate
    hashmap['history'] = history

