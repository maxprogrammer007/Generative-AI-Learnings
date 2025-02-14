from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


document  = [
    "Delhi is the capital of India",
    "Delhi is the capital of India. It is a large city in the north of India.",
    "Delhi is known for its historical monuments and vibrant culture."
]
result = embedding.embed_documents(document)
print("Embedding for documents:")
print(str(result))
print("Embedding length:", len(result))
print("Embedding type:", type(result))