import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community import embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

urls = ["https://www.highsignal.io/blog/best-saas-boilerplates-for-founders/", "https://www.sitepoint.com/saas-boilerplate-intro/"]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list) 

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model='mxbai-embed-large'),
)
retriever = vectorstore.as_retriever()

def create_rag_chain(retriever):
    rag_template = """Give the names of the boilerplates that contain the names of the technologies named in the question based on the following context only::
    {context}
    Question: {question}"""

    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | ChatOllama(model="llama3.1")
    )

    return rag_chain

def run_rag_chain_loop(retriever):
    rag_chain = create_rag_chain(retriever)
    
    while True:
        user_input = input("Type your question (or '/exit' to finish): ")
        
        if user_input.lower() == '/exit':
            print("Thank you for using the system, see you later!")
            break
        
        response = rag_chain.invoke(user_input)
        print(f"Response: {response}")

run_rag_chain_loop(retriever)