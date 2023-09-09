#!/bin/env python 
# inference.py 

from __future__ import print_function, absolute_import, annotations 
from tpying import List, Dict, Set 
import sys 
import os 
import io
import json 
import pickle
import joblib 
import pandas as pd 
from datetime import datetime 
from sentence_transformers import SentenceTransformer 
from sagemaker_inference import encoder, decoder 

MODEL_DIR = 'all-mpnet-base-v2-finetuned'

def model_fn(mode_dir: str=MODEL_DIR):
  """
  Load the fine-tuned embedding model.
  Return: model
  """
  print('--', os.listdir(os.environ["SM_MODEL_DIR"]), '--')
  print('**', os.path.join(os.environ["SM_MODEL_DIR"], MODEL_DIR), '**')

  embedding_model = SentenceTransformer(os.path.join(environ["SM_MODEL_DIR"], MODEL_DIR))
  return embedding_model

def input_fn(input_data, content_type):
  if isinstance(input_data, str):
    json_object = json.loads(input_data)
    key = [k for k in json_object.keys()][0]
    input_data = json_object[key]
  else:
    print('Input bytes type:', type(input_data))
    key = [k for k in input_data.keys()][0]
    input_data = input_data[key]
  return input_data 

def predict_fn(input_data, embedding_model, content_type=None, accept=None):
  embeds = embedding_model.encode(input_data)
  print(embeds.shape)
  return embeds.tolist()

def ouput_fn(embeddings, content_type):
  output = json.dumps(embeddings)
  return output


if __name__ == '__main__':
  model_dir = 'all-mpnet-base-v2-finetuned'
  user_query = '..........................'
  input_data = input_fn(user_query, 'application/json')
  embedding_model = model_fn(model_dir)
  embeds = predict_fn(input_data, embedding_model)
  output = output_fn(embeds, 'application/json')
  print('output=', output)
  print('output type=', type(output))
