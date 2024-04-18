import openai

from typing import List, Iterator
import pandas as pd
import numpy as np
import os
from ast import literal_eval

from pymilvus import connections
from pymilvus import utility, Collection, FieldSchema, CollectionSchema, DataType


openai.api_base = 'http://172.20.193.39/v1'

# I've set this to our new embeddings model, this can be changed to the embedding model of your choice
EMBEDDING_MODEL = "text-embedding-ada-002"

# Ignore unclosed SSL socket warnings - optional in case you get these errors
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

article_df = pd.read_csv("500.csv")

print("head")
print(article_df.head())

# Read vectors from strings back into a list
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Set vector_id to be a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

print("info")
print(article_df.info(show_counts=True))


connections.connect(host='localhost', port=19530)  # Local instance defaults to port 19530
# Remove the collection if it already exists.
if utility.has_collection('articles'):
    utility.drop_collection('articles')

fields = [
    FieldSchema(name='id', dtype=DataType.INT64),
    FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=1000),  # Strings have to specify a max length [1, 65535]
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='content_vector', dtype=DataType.FLOAT_VECTOR, dim=len(article_df['content_vector'][0])),
    FieldSchema(name='vector_id', dtype=DataType.INT64, is_primary=True, auto_id=False),
]

col_schema = CollectionSchema(fields)

col = Collection('articles', col_schema)

# Using a basic HNSW index for this example
index = {
    'index_type': 'HNSW',
    'metric_type': 'L2',
    'params': {
        'M': 8,
        'efConstruction': 64
    },
}

col.create_index('content_vector', index)
col.load()


# Using the above provided batching function from Pinecone
def to_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    splits = df.shape[0] / batch_size
    if splits <= 1:
        yield df
    else:
        for chunk in np.array_split(df, splits):
            yield chunk

# Since we are storing the text within Milvus we need to clip any that are over our set limit.
# We can also set the limit to be higher, but that slows down the search requests as more info 
# needs to be sent back.
def shorten_text(text):
    if len(text) >= 996:
        return text[:996] + '...'
    else:
        return text

for batch in to_batches(article_df, 1000):
    batch = batch.drop(columns = ['title_vector'])
    batch['text'] = batch.text.apply(shorten_text)
    # Due to the vector_id being converted to a string for compatiblity for other vector dbs,
    # we want to swap it back to its original form.
    batch['vector_id'] = batch.vector_id.apply(int)
    col.insert(batch)

col.flush()

openai.api_key = os.getenv("OPENAI_API_KEY", "missing_key")

def query_article(query, top_k=5):
    # Generate the embedding with openai
    embedded_query = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )["data"][0]['embedding']

    # Using some basic params for HNSW
    search_param = {
        'metric_type': 'L2',
        'params': {
            'ef': max(64, top_k)
        }
    }

    # Perform the search.
    res = col.search([embedded_query], 'content_vector', search_param, output_fields = ['title', 'url'], limit = top_k)

    ret = []
    for hit in res[0]:
        # Get the id, distance, and title for the results
        ret.append({'vector_id': hit.id, 'distance': hit.score, 'title': hit.entity.get('title'), 'url': hit.entity.get('url')})
    return ret


for x in query_article('fastest plane ever made', 3):
    print(x.items())
