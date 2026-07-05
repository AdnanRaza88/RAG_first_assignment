import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF Chatbot with Groq")

with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input("GROQ API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if st.button("🚀 Process PDF") and uploaded_file and groq_api_key:
        with st.spinner("Processing..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local("faiss_index")
            
            st.session_state.vectorstore_ready = True
            st.success(f"✅ Done! {len(docs)} chunks created")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything from your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore_ready" not in st.session_state:
        st.error("Please upload and process PDF first")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
                result = qa_chain({"query": prompt})
                answer = result["result"]
                
                st.markdown(answer)
                
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"**Chunk {i+1}**")
                        st.write(doc.page_content[:400] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
