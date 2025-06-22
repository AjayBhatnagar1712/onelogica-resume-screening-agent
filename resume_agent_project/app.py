import os
import streamlit as st
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import pipeline

# Constants
RESUME_FOLDER = r"D:\work\resume_agent_project\data\resumes"

# Load and embed once
@st.cache_resource
def load_and_embed_resumes(folder_path: str) -> FAISS:
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["source"] = filename
            docs.extend(pdf_docs)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore

# Search top-k matching resumes
def search_resumes(vectorstore: FAISS, job_description: str, k: int = 3) -> List[Tuple[Document, float]]:
    results = vectorstore.similarity_search_with_score(job_description, k=k*2)
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

# Summarization with caching
@st.cache_data
def summarize_text(text: str) -> str:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    input_text = text[:3000]  # Truncate long resumes
    summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

# Streamlit App
def main():
    st.title("🔍 Fast Resume Matching App")

    job_description = st.text_area("Enter Job Description", height=200)
    if st.button("Match Resumes") and job_description.strip():
        with st.spinner("Loading and matching resumes..."):
            vectorstore = load_and_embed_resumes(RESUME_FOLDER)
            results = search_resumes(vectorstore, job_description)

        st.subheader("📄 Top Matching Resumes:")
        for rank, (doc, score) in enumerate(results, start=1):
            st.markdown(f"**Rank {rank}: {doc.metadata.get('source')}**")
            st.write(f"**Match Score:** {score:.2f}")
            st.write(doc.page_content[:400] + "...")

            explanation = summarize_text(doc.page_content)
            st.success(f"**Why this matches:** {explanation}")
    else:
        st.info("Please enter a job description to start matching.")

if __name__ == "__main__":
    main()
