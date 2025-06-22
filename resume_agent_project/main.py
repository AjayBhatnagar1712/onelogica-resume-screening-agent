import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def load_resumes(folder_path: str) -> List[Document]:
    """
    Load and return all PDFs from the resume folder as LangChain Documents.
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()
            for d in loaded_docs:
                d.metadata["source"] = filename  # track filename
            docs.extend(loaded_docs)
    return docs

def embed_resumes(docs: List[Document]) -> FAISS:
    """
    Create and return a FAISS vectorstore of the resumes.
    """
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore

def search_resumes(vectorstore: FAISS, job_description: str, k: int = 3) -> List[Tuple[Document, float]]:
    """
    Search for top-k matching resumes for the given job description.
    Returns a list of tuples: (Document, similarity score)
    """
    results = vectorstore.similarity_search_with_score(job_description, k=k*2)  # get more, then filter
    seen = set()
    unique_results = []
    for doc, score in results:
        source = doc.metadata.get("source")
        if source not in seen:
            seen.add(source)
            unique_results.append((doc, score))
        if len(unique_results) == k:
            break
    return unique_results
