from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

    
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05")

result = llm.invoke("What is the capital of India?")
print(result.content)