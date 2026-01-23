from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema import Document
import os

def task1_document_loading():
    print("=" * 50)
    print("TASK 1: Document Loading")
    print("=" * 50)
    
    loader = TextLoader("data.txt")
    documents = loader.load()
    
    print(f"Number of documents loaded: {len(documents)}")
    print(documents[0].page_content[:200] + "...")
    print(f"Document metadata: {documents[0].metadata}")
    
    return documents

def task2_text_splitting(documents):
    print("\n" + "=" * 50)
    print("TASK 2: Text Splitting")
    print("=" * 50)
    
    splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separator="\n"
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"Original document length: {len(documents[0].page_content)} characters")
    print(f"Number of chunks created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"Content: {chunk.page_content}")
        print(f"Length: {len(chunk.page_content)} characters")
    
    return chunks

def task3_embeddings_creation(chunks):
    print("\n" + "=" * 50)
    print("TASK 3: Creating Embeddings")
    print("=" * 50)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    sample_text = "LangChain is a framework for AI applications"
    embedding_vector = embeddings.embed_query(sample_text)
    
    print(f"Sample text: {sample_text}")
    print(f"Embedding dimension: {len(embedding_vector)}")
    print(f"First 5 embedding values: {embedding_vector[:5]}")
    
    return embeddings

def task4_vector_store(chunks, embeddings):
    print("\n" + "=" * 50)
    print("TASK 4: Vector Store Operations")
    print("=" * 50)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print("Vector store created successfully!")
    print(f"Number of documents in vector store: {vectorstore.index.ntotal}")
    
    query = "What is LangChain?"
    results = vectorstore.similarity_search(query, k=2)
    
    print(f"\nQuery: {query}")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {result.page_content}")
    
    return vectorstore

def task5_document_search_demo(vectorstore):
    print("\n" + "=" * 50)
    print("TASK 6: Document Search Demo")
    print("=" * 50)
    
    queries = [
        "What are the key features of LangChain?",
        "How is LangChain used for chatbots?",
        "What is document retrieval?",
        "Tell me about vector stores"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = vectorstore.similarity_search(query, k=1)
        if results:
            print(f"Answer: {results[0].page_content}")
        else:
            print("No relevant information found.")

def task6_memory_simulation():
    print("\n" + "=" * 50)
    print("TASK 7: Conversation Memory Simulation")
    print("=" * 50)
    
    conversation_history = []
    
    def add_to_memory(user_input, ai_response):
        conversation_history.append({
            "user": user_input,
            "ai": ai_response,
            "timestamp": "simulated"
        })
    
    def get_conversation_context():
        print()
        context = ""
        for exchange in conversation_history[-3:]:
            context += f"User: {exchange['user']}\nAI: {exchange['ai']}\n\n"
        return context
    
    exchanges = [
        ("What is LangChain?", "LangChain is a framework for building AI applications."),
        ("What can it do?", "It helps with document processing, embeddings, and AI chains."),
        ("Give me an example", "You can build chatbots and Q&A systems with it.")
    ]
    
    for user_msg, ai_msg in exchanges:
        add_to_memory(user_msg, ai_msg)
    
    print("Conversation Context (last 3 exchanges):")
    print(get_conversation_context())


def main():
    print("🚀 LangChain Basic Tasks Demo")
    print("This demo shows fundamental LangChain operations\n")
    
    try:
        # Task 1: Document Loading
        documents = task1_document_loading()
        
        # # Task 2: Text Splitting
        # chunks = task2_text_splitting(documents)
        
        # # Task 3: Embeddings
        # embeddings = task3_embeddings_creation(chunks)
        
        # # Task 4: Vector Store
        # vectorstore = task4_vector_store(chunks, embeddings)
        
        # # Task 5: Document Search
        # task5_document_search_demo(vectorstore)
        
        # # Task 6: Memory Simulation
        # task6_memory_simulation()
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    main()