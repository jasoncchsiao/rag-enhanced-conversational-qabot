import yaml
import logging 
import argparse 
from modules.preprocessor import *
from modules.index_db import *
from modules.retriever import *
from modules.reader import *
from modules.metrics import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='app commandline')
  parser.add_argument('--log_level', type=str, default='INFO')
  args = parser.parse_args()

  FORMAT = '%(asctime)-15s %(message)s'
  logging.basicConfig(format=FORMAT, level.args.log_level)
  logger = logging.getLogger('global_logger')

dir_path = 'data/source_files'
embedding_model = 'modules/models/all-mpnet-base-v2-finetuned'
embedding_path = 'data/existing_embedding_table.csv'
top_k = 5

d = EMBEDDING_DIMENSION
m = 8; nbits = 8

retriever = Retriever(embedding_path, 
                      top_k, dir_path, embedding_model,
                      d, m, nbits)
filenames, chunks = retriever.pp.get_chunks()
embeddings = retriever.pp.get_embeddings(chunks)
retriever.update_indexDB(embeddings)
embeddings_df = retriever.get_lookup_df(embeddings, filenames, chunks)
user_query = '..........'
uq_embeds = retriever.get_user_embeddings(user_query)
context = retriever.get_context_from_retriever(uq_embeds, embeddings_df)
print('CONTEXT=', context)
original_prompt = \
"""YOU ARE AN ASSISTANT."""
summarizer_model = \
'modules/models/summarizer/t5-small'
summarizer_tokenizer = \
'modules/models/summarizer/t5-small/tokenizer'
llm_tokenizer = \
'modules/models/llm/tiiuae/falcon-40b-instruct'
qabot = Reader(original_prompt,
               summarizer_model,
               summarizer_tokenizer,
               llm_tokenizer)
print('FIRST MESSAGE from llm: ',
      '\n------------------------------------------------------------',
      qabot.query_llm("""\nUser: {} \nAssistant:""".format(user_query),
                      """{}""".format(context)),
      '\n-----------------------------------------------------------')
new_uq = 'Can you elaborate more?'
user_query += ' ' + new_uq 
uq_embeds = retriever.get_user_embeddings(user_query)
context = retriever.get_context_from_retriever(uq_embeds, embeddings_df)
print("SECOND MESSAGE from llm: ',
      '\n-----------------------------------------------------------',
      qabot.query_llm("""\nUser: {} \nAssistant:""".format(new_uq),
                      """{}""".format(context)),
      '\n-----------------------------------------------------------')
print('CHAT_HISTORY=', qabot.prompt)
print('current messages=', qabot.messages)
