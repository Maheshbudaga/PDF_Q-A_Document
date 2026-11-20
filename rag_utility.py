import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

#load environment variables from .env files
load_dotenv()


working_dir = os.path.dirname(os.path.abspath((__file__)))

# load the embedding model
embedding = HuggingFaceEmbeddings()

#load the groq model from groq
llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0
)

#document ingestion
def process_document_to_chroma_db(file_name):
   
    file_path = os.path.join(working_dir, file_name)

    # Load PDF
    loader = PyPDFLoader(file_path)

    documents = loader.load()
    #split the text into chunks for embeding

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    #store the document chunks in a chroma vector database

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )

    return 0



def answer_question(user_question):

    #load the persistent chroma vector database
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    print("Total documents :", vectordb._collection.count())

    #create a retriever for document search

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(user_question)

    print("Retrieved documents:", len(docs))

    for doc in docs:
        print(doc.page_content)

    #create a RetrieverQA chain to  answer user questons using llama-3.3-70b
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    ) 
    response = qa_chain.invoke({"query": user_question})

    answer = response["result"]

    print("Retrieved sources:", response["source_documents"])

    return answer






 