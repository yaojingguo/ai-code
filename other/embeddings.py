import tiktoken

from openai.embeddings_utils import get_embedding

openai.api_base = "http://172.20.193.39/v1"
openai.api_key = "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"

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
