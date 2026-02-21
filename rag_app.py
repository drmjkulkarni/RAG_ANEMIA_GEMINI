import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Securely retrieve the key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Critical Error: OPENAI_API_KEY is missing. Check your .env file.")

os.environ["OPENAI_API_KEY"] = openai_api_key
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def create_and_run_rag(pdf_path, question):
    # 1. Load the PDF Document
    if not os.path.exists(pdf_path):
        return f"Error: The file {pdf_path} was not found in the directory."
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split text into chunks for accurate retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # 3. Create a Vector Store for document embeddings
    # FAISS is used here for local, rapid vector similarity search
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    
    # Retrieve the top 4 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 

    # 4. Define the strict instruction prompt
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

    # 5. Initialize the LLM with temperature 0 to enforce strict, deterministic outputs
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Helper function to format retrieved documents into a single string
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 6. Construct the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Execute the chain
    return rag_chain.invoke(question)

if __name__ == "__main__":
    pdf_filename = "KDIGO_2026_ANEMIA.pdf"
    
    # Example 1: In-scope question (Will trigger extraction and quotes)
    question_1 = "What should be done if an active infection is present during iron therapy?"
    print(f"User: {question_1}")
    print(f"RAG: {create_and_run_rag(pdf_filename, question_1)}\n")
    
    # Example 2: Out-of-scope question (Will trigger the exact rejection phrase)
    question_2 = "What are the recommended treatments for type 2 diabetes?"
    print(f"User: {question_2}")
    print(f"RAG: {create_and_run_rag(pdf_filename, question_2)}")