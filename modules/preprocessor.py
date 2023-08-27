from __future__ import annotations
import yaml
import logging
import argparse
import os
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict
import re
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
EMBEDDING_DIMENSION = 768
TOKENIZER_MODEL = 'sentence-transformers/all-mpnet-base-v2'
MAX_TOKENS = 380
STRIDES = 50

class Preprocessor:
    def __init__(self, dir_path: str, embedding_model: str=EMBEDDING_MODEL,
                 tokenizer_model: str=TOKENIZER_MODEL, max_token_length: int=MAX_TOKENS,
                 strides: int=STRIDES) -> None:
        self.dir_path = dir_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tokenizer_model = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_token_length = max_token_length
        self.strides = strides

    def scan_kb_by_titles(self, testing: bool=True) -> List[str, str]:
        filenames = []
        texts = []
        for i, (root, dirs, files) in enumerate(os.walk(self.dir_path)):
            if i == 0:
                continue
            for _, name in enumerate(files):
                file_name = os.path.join(root, name)
                with open(file_name, 'r', encoding='utf-8') as f:
                    reader = f.readlines()
                text = [self._remove_newlines(line.strip()).strip() for line in reader]
                text = ' '.join(text)
                filenames.append(file_name)
                texts.append(text)
        if testing:
            return filenames[-300:], texts[-300:]
        else:
            return filenames, texts

    def get_tokenized_texts(self, filenames: List[str], texts: List[str], is_langchain: bool=True) -> List[str, str]:
        idx2filename = {i: filename for i, filename in enumerate(filenames)}
        #################### LANGCHAIN SECTION ##################################
        if is_langchain:
            texts = [re.sub(r"<.*?.*?>","", text) for text in texts]
            texts = [self._remove_newlines(text.strip()).strip() for text in texts]
            text_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=self.strides,
                model_name=self.embedding_model,
                tokens_per_chunk=self.max_token_length
            )
            all_chunks = []
            updated_filenames = []
            for i in range(len(filenames)):
                chunks = text_splitter.split_text(text=texts[i])
                all_chunks.append(chunks)
                n = len(chunks)
                updated_filenames.extend([idx2filename[i] for _ in range(n)])
            return updated_filenames, all_chunks
        else:
            tokenized_texts = \
            self.tokenizer_model(
                texts,
                max_length=self.max_token_length,
                truncation=True,
                stride=self.strides,
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
            # DECODE BACK TO TEXT
            chunks = []
            updated_filenames = []
            n = len(tokenized_texts['input_ids'])
            for i in range(n):
                chunk = self.tokenizer_model.decode(tokenized_texts['input_ids'][i])[len('<s>')+1: -len('</s>')]
                chunk = re.sub(r"<.*?.*?>","", chunk)
                chunk = self._remove_newlines(chunk.strip()).strip()
                filename = idx2filename[tokenized_texts['overflow_to_sample_mapping'][i]]
                if self._keep_chunk(chunk):
                    chunks.append(chunk)
                    updated_filenames.append(filename)
            return updated_filenames, chunks

    def _keep_chunk(self, chunk: str) -> bool:
        words = chunk.split()
        return False if len(words) < 20 else True

    def _remove_newlines(self, text: str) -> str:
        text = text.replace('\n', ' ')
        text = text.replace('\\n', ' ')
        text = text.replace('|', ' ')
        text = text.replace('-', ' ')
        while text.find('  ') != -1:
            text = text.replace('  ', ' ')
        return text

    def get_embeddings                          (self, chunks: List[str]) -> np.ndarray:
        embeds = self.embedding_model.encode(chunks)
        print(embeds.shape)
        return embeds

    def write_embedding_to_csv(self, filename: List[str], chunks: List[str], embeddings: np.ndarray,
                               output_path: str) -> None:
        df = pd.DataFrame({
            'idx': range(embeddings.shape[0]),
            'filename': filenames,
            'original_text': chunks,
            'embedding': [embed for embed in embeddings]
        })
        df.to_csv(output_path, index=False)

if __name__ == '__main__':
    dir_path = '../../notebooks/vector-search/kb_extract'
    pp = Preprocessor(dir_path=dir_path)
    filenames, texts = pp.scan_kb_by_titles()
    print(texts.__len__())
    filenames, chunks = pp.get_tokenized_texts(filenames, texts)
    print(chunks.__len__())
    print('CHUNKS from langchain...')
    print(chunks[-5:])
    embeddings = pp.get_embedding(chunks)
    pp2 = Preprocessor(dir_path=dir_path)
    filenames, texts = pp2.scan_by_titles()
    print(texts.__len__())
    filenames, chunks = pp2.get_tokenized_texts(filenames, texts, False)
    print(chunks[-5:])
    embeddings = pp2.get_embeddings(chunks)
    pp2.write_embedding_to_csv(filenames, chunks, embeddings, '../data/embeddings_tf.csv')
    print('Done writing embeddings from tf...')
