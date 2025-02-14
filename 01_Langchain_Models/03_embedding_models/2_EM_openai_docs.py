from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Delhi is the capital of India",
    "India is a country in South Asia",
    "The capital of France is Paris",
]

result = OpenAIEmbeddings.embed_documents(documents)
print(str(result))