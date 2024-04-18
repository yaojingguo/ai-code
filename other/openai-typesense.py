import openai

from typing import List, Iterator
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval

import typesense

openai.api_base = 'http://172.20.193.39/v1'

EMBEDDING_MODEL = "text-embedding-ada-002"

article_df = pd.read_csv('500.csv')
article_df.head()

article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)
article_df['vector_id'] = article_df['vector_id'].apply(str)

article_df.info(show_counts=True)

import typesense

typesense_client = \
    typesense.Client({
        "nodes": [{
            "host": "localhost",  # For Typesense Cloud use xxx.a1.typesense.net
            "port": "8108",       # For Typesense Cloud use 443
            "protocol": "http"    # For Typesense Cloud use https
          }],
          "api_key": "xyz",
          "connection_timeout_seconds": 60
        })


try:
    typesense_client.collections['wikipedia_articles'].delete()
except Exception as e:
    pass


schema = {
    "name": "wikipedia_articles",
    "fields": [
        {
            "name": "content_vector",
            "type": "float[]",
            "num_dim": len(article_df['content_vector'][0])
        },
        {
            "name": "title_vector",
            "type": "float[]",
            "num_dim": len(article_df['title_vector'][0])
        }
    ]
}


create_response = typesense_client.collections.create(schema)
print(create_response)


document_counter = 0
documents_batch = []

for k,v in article_df.iterrows():
    # Create a document with the vector data
    # Notice how you can add any fields that you haven't added to the schema to the document.
    # These will be stored on disk and returned when the document is a hit.
    # This is useful to store attributes required for display purposes.
    document = {
        "title_vector": v["title_vector"],
        "content_vector": v["content_vector"],
        "title": v["title"],
        "content": v["text"],
    }
    documents_batch.append(document)
    document_counter = document_counter + 1
    # Upsert a batch of 100 documents
    if document_counter % 100 == 0 or document_counter == len(article_df):
        response = typesense_client.collections['wikipedia_articles'].documents.import_(documents_batch)
        # print(response)

        documents_batch = []
        print(f"Processed {document_counter} / {len(article_df)} ")

print(f"Imported ({len(article_df)}) articles.")


def query_typesense(query, field='title', top_k=20):
    # Creates embedding vector from user query
    openai.api_key = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    embedded_query = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )['data'][0]['embedding']
    typesense_results = typesense_client.multi_search.perform({
        "searches": [{
            "q": "*",
            "collection": "wikipedia_articles",
            "vector_query": f"{field}_vector:([{','.join(str(v) for v in embedded_query)}], k:{top_k})"
        }]
    }, {})
    return typesense_results


query_results = query_typesense('modern art in Europe', 'title')
for i, hit in enumerate(query_results['results'][0]['hits']):
    document = hit["document"]
    vector_distance = hit["vector_distance"]
    print(f'{i + 1}. {document["title"]} (Distance: {vector_distance})')


query_results = query_typesense('Famous battles in Scottish history', 'content')
for i, hit in enumerate(query_results['results'][0]['hits']):
    document = hit["document"]
    vector_distance = hit["vector_distance"]
    print(f'{i + 1}. {document["title"]} (Distance: {vector_distance})')
