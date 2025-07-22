import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from duckduckgo_search import DDGS
from transformers import pipeline
import os
import tempfile
import json

# 🧠 Use a local LLM (e.g., distilbert-based)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# 📘 Streamlit UI
st.title("📚 Intelligent PDF + Web RAG Assistant (No API Key)")
st.markdown("Upload a PDF and ask questions. If the answer isn’t in the PDF, I’ll search the web for you!")

uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")
query = st.text_input("❓ Enter your question")

if uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Search PDF content
    results = retriever.get_relevant_documents(query)
    context_text = " ".join([doc.page_content for doc in results])

    # Context Evaluation Prompt (simple heuristic-based)
    def evaluate_context(context):
        relevance = 1 if query.lower() in context.lower() else 0.5
        completeness = 0.7 if len(context.split()) > 50 else 0.4
        accuracy = 0.8  # Assuming retrieved PDF is mostly factual
        specificity = 0.6 if len(context.split()) > 30 else 0.3
        overall = "EXCELLENT" if relevance > 0.8 and completeness > 0.6 else "FAIR"
        return {
            "Relevance": relevance,
            "Completeness": completeness,
            "Accuracy": accuracy,
            "Specificity": specificity,
            "Overall": overall
        }

    eval_json = evaluate_context(context_text)
    st.subheader("🔍 Context Evaluation")
    st.json(eval_json)

    # Decision Branch
    if eval_json["Overall"] in ["POOR", "FAIR"]:
        st.warning("⚠ Context was insufficient. Retrieving from web...")
        with DDGS() as ddgs:
            web_results = [r for r in ddgs.text(query, max_results=3)]
            web_context = "\n".join([r["body"] for r in web_results if "body" in r])
            sources = ", ".join([r["href"] for r in web_results if "href" in r])

        # Get final answer from web context
        result = qa_pipeline(question=query, context=web_context)
        final_answer = f"""
🔍 Context Quality: FAIR  
📊 Confidence Level: MEDIUM  
🎯Answer: {result['answer']}  
📚 Sources: {sources}  
⚠ Note: Initial retrieval was insufficient. Answer based on corrected retrieval.
"""
    else:
        # Use PDF content
        result = qa_pipeline(question=query, context=context_text)
        final_answer = f"""
🔍 Context Quality: {eval_json["Overall"]}  
📊 Confidence Level: HIGH  
🎯Answer: {result['answer']}  
📚 Sources: PDF
"""

    st.subheader("📤 Final Answer")
    st.markdown(final_answer)
