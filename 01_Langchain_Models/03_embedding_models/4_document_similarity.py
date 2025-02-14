from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# Initialize OpenAIEmbeddings with the desired model and dimensions
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    " sachin is best batsman",
    " sachin is best player",
    "virat is god of cricket",
    "virat is good player",
]

# Embed the documents

embedded_docs = openai_embeddings.embed_documents(documents)
print("Embedded documents:", embedded_docs)

query = " tell me about sachin"
# Embed the query
embedded_query = openai_embeddings.embed_query(query)



# Calculate cosine similarity between the query and each document
SCORES = cosine_similarity([embedded_query], embedded_docs)[0]
index , score = sorted(list(enumerate(SCORES)),key=lambda x: x[1])[-1]
# Sort the documents based on similarity scores
print(query)
print("Most similar document index:", index)
print("Most similar document score:", score)

