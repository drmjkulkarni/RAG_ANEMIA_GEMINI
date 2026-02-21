import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Set your OpenAI API Key securely in Streamlit's secrets management, NOT hardcoded.
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def initialize_rag(pdf_path):
    if not os.path.exists(pdf_path):
        st.error(f"Error: The file {pdf_path} was not found.")
        st.stop()
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 1. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 2. Setup Sparse Retriever (Keyword Search)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 4  # Retrieve top 4 exact keyword matches

    # 3. Setup Dense Retriever (Semantic Vector Search)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 

    # 4. Combine them into a Hybrid Ensemble Retriever
    # The weights parameter balances the importance of keyword vs. semantic search (0.5 each is standard)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5] 
    )

    # ... (Your system_prompt and LLM definition remain exactly the same) ...
    system_prompt = """You are an expert medical assistant. Your sole purpose is to answer the user's questions based STRICTLY and ONLY on the provided context.

    Rules you must follow:
    1. If the answer is found in the context, you MUST answer the question and include exact quotes from the text, placing the quotes inside quotation marks ("").
    2. If the context does not contain the specific information needed to answer the question, you MUST reply EXACTLY with: "This is beyond the scope of the document."
    3. Do not rely on outside knowledge, make assumptions, or attempt to be helpful if the information is missing.

    Context:
    {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # 5. Update the RAG chain to use the ensemble_retriever instead of the basic retriever
    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    return rag_chain

# --- Streamlit UI ---
st.title("KDIGO 2026 Anemia Guidelines Q&A")
st.warning("This system strictly answers from the provided KDIGO document. Out-of-scope queries will be rejected.")

pdf_filename = "KDIGO_2026_ANEMIA.pdf"
rag_chain = initialize_rag(pdf_filename)

user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_question:
        with st.spinner("Analyzing document..."):
            response = rag_chain.invoke(user_question)
            st.write("### Answer")
            st.write(response)
    else:
        st.error("Please enter a question first.")