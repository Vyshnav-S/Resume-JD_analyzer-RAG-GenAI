import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load job dataset
df = pd.read_csv("data/jobs.csv")

documents = []
for _, row in df.iterrows():
    content = f"""
Job Title: {row['Job Title']}
Job Description: {row['Job Description']}
"""
    documents.append(Document(page_content=content))

# Free embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS index
vectorstore = FAISS.from_documents(documents, embedding_model)

# Save FAISS index
vectorstore.save_local("vectorstore")

print("✅ FAISS index created successfully")

