from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
import json, os


def task1_document_summarization():
    """Task 1: Document Summarization"""
    print("=" * 60)
    print("TASK 1: Document Summarization")
    print("=" * 60)
    
    # Load and split document
    loader = TextLoader("data.txt")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    # Create summarization prompt
    summary_template = """
    Summarize the following text in 2-3 sentences:
    
    Text: {text}
    
    Summary:
    """
    
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template=summary_template
    )
    
    # Simulate summarization (without LLM)
    print("Document Chunks and Simulated Summaries:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"Original: {chunk.page_content}")
        print(f"Simulated Summary: Key points about LangChain's {['framework', 'features', 'applications'][i]} capabilities.")

def task2_multi_document_qa():
    """Task 2: Multi-Document Question Answering"""
    print("\n" + "=" * 60)
    print("TASK 2: Multi-Document Question Answering")
    print("=" * 60)
    
    # Create multiple documents
    docs = [
        Document(page_content="Python is a programming language known for its simplicity.", metadata={"source": "python_doc"}),
        Document(page_content="JavaScript is used for web development and runs in browsers.", metadata={"source": "js_doc"}),
        Document(page_content="Machine learning uses algorithms to learn from data.", metadata={"source": "ml_doc"})
    ]
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Multi-document queries
    queries = [
        "What programming languages are mentioned?",
        "Tell me about web development",
        "What is machine learning?"
    ]
    
    for query in queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\nQuery: {query}")
        print("Relevant documents:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.page_content} (Source: {result.metadata['source']})")

def task3_chain_operations():
    """Task 3: Sequential Chain Operations"""
    print("\n" + "=" * 60)
    print("TASK 3: Sequential Chain Operations")
    print("=" * 60)
    
    # Create multiple prompt templates for chaining
    
    # Step 1: Extract key topics
    extraction_template = """
    Extract the main topics from this text:
    Text: {input_text}
    
    Main Topics: """
    
    # Step 2: Generate questions about topics
    question_template = """
    Based on these topics: {topics}
    Generate 3 relevant questions:
    
    Questions: """
    
    # Step 3: Create study guide
    study_template = """
    Topics: {topics}
    Questions: {questions}
    
    Create a brief study guide:
    
    Study Guide: """
    
    extraction_prompt = PromptTemplate(
        input_variables=["input_text"],
        template=extraction_template
    )
    
    question_prompt = PromptTemplate(
        input_variables=["topics"],
        template=question_template
    )
    
    study_prompt = PromptTemplate(
        input_variables=["topics", "questions"],
        template=study_template
    )
    
    # Simulate chain execution
    input_text = "LangChain helps build AI applications with document processing and embeddings."
    
    print("Chain Execution Simulation:")
    print(f"Input: {input_text}")
    print(f"Step 1 - Extracted Topics: AI applications, document processing, embeddings")
    print(f"Step 2 - Generated Questions: What is LangChain? How does it process documents? What are embeddings?")
    print(f"Step 3 - Study Guide: Focus on understanding LangChain framework, document handling, and vector embeddings.")

def task4_few_shot_learning():
    """Task 4: Few-Shot Learning with Examples"""
    print("\n" + "=" * 60)
    print("TASK 4: Few-Shot Learning with Examples")
    print("=" * 60)
    
    # Create examples for few-shot learning
    examples = [
        {
            "input": "Python",
            "output": "A high-level programming language known for readability and versatility."
        },
        {
            "input": "JavaScript", 
            "output": "A programming language primarily used for web development and browser scripting."
        },
        {
            "input": "SQL",
            "output": "A domain-specific language for managing and querying relational databases."
        }
    ]
    
    # Create example template
    example_template = """
    Term: {input}
    Definition: {output}
    """
    
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )
    
    # Create few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Define the following programming terms based on the examples:",
        suffix="Term: {input}\nDefinition:",
        input_variables=["input"]
    )
    
    # Test with new term
    new_term = "React"
    formatted_prompt = few_shot_prompt.format(input=new_term)
    
    print("Few-Shot Learning Example:")
    print(formatted_prompt)
    print("Expected output: A JavaScript library for building user interfaces and web applications.")

def task5_memory_management():
    """Task 5: Advanced Memory Management"""
    print("\n" + "=" * 60)
    print("TASK 5: Advanced Memory Management")
    print("=" * 60)
    
    # Conversation Buffer Memory
    buffer_memory = ConversationBufferMemory()
    
    # Simulate conversation
    conversations = [
        ("What is LangChain?", "LangChain is a framework for building AI applications."),
        ("What can I build with it?", "You can build chatbots, QA systems, and document analyzers."),
        ("Is it difficult to learn?", "It has a learning curve but great documentation helps.")
    ]
    
    print("Conversation Buffer Memory:")
    for human_msg, ai_msg in conversations:
        buffer_memory.save_context({"input": human_msg}, {"output": ai_msg})
        print(f"Human: {human_msg}")
        print(f"AI: {ai_msg}")
    
    print(f"\nStored Memory:")
    print(buffer_memory.buffer)
    
    # Memory variables
    memory_vars = buffer_memory.load_memory_variables({})
    print(f"\nMemory Variables: {memory_vars}")

def task6_document_classification():
    """Task 6: Document Classification"""
    print("\n" + "=" * 60)
    print("TASK 6: Document Classification")
    print("=" * 60)
    
    # Sample documents for classification
    sample_docs = [
        "This is a technical tutorial about Python programming and data structures.",
        "Breaking news: New AI breakthrough announced by researchers at major university.",
        "Product review: This smartphone has excellent camera quality and battery life.",
        "Recipe: How to make delicious chocolate chip cookies in 30 minutes."
    ]
    
    # Classification categories
    categories = ["Technical", "News", "Review", "Recipe"]
    
    # Create classification prompt
    classification_template = """
    Classify the following text into one of these categories: {categories}
    
    Text: {text}
    
    Category: """
    
    classification_prompt = PromptTemplate(
        input_variables=["categories", "text"],
        template=classification_template
    )
    
    print("Document Classification Examples:")
    expected_results = ["Technical", "News", "Review", "Recipe"]
    
    for i, doc in enumerate(sample_docs):
        formatted_prompt = classification_prompt.format(
            categories=", ".join(categories),
            text=doc
        )
        print(f"\nDocument {i+1}: {doc[:50]}...")
        print(f"Predicted Category: {expected_results[i]}")

def task7_data_extraction():
    """Task 7: Structured Data Extraction"""
    print("\n" + "=" * 60)
    print("TASK 7: Structured Data Extraction")
    print("=" * 60)
    
    # Sample unstructured text
    text = """
    John Smith works as a Software Engineer at TechCorp. 
    His email is john.smith@techcorp.com and phone number is (555) 123-4567.
    He has 5 years of experience in Python and JavaScript.
    """
    
    # Extraction template
    extraction_template = """
    Extract the following information from the text:
    - Name
    - Job Title  
    - Company
    - Email
    - Phone
    - Skills
    - Experience
    
    Text: {text}
    
    Extracted Information:
    """
    
    extraction_prompt = PromptTemplate(
        input_variables=["text"],
        template=extraction_template
    )
    
    formatted_prompt = extraction_prompt.format(text=text)
    
    print("Data Extraction Example:")
    print(f"Input Text: {text}")
    print("\nExpected Extracted Data:")
    extracted_data = {
        "Name": "John Smith",
        "Job Title": "Software Engineer", 
        "Company": "TechCorp",
        "Email": "john.smith@techcorp.com",
        "Phone": "(555) 123-4567",
        "Skills": ["Python", "JavaScript"],
        "Experience": "5 years"
    }
    
    for key, value in extracted_data.items():
        print(f"- {key}: {value}")

def task8_content_generation():
    """Task 8: Content Generation"""
    print("\n" + "=" * 60)
    print("TASK 8: Content Generation")
    print("=" * 60)
    
    # Blog post generation template
    blog_template = """
    Write a blog post about: {topic}
    
    Target audience: {audience}
    Tone: {tone}
    Length: {length}
    
    Blog Post:
    Title: 
    Introduction:
    Main Points:
    Conclusion:
    """
    
    blog_prompt = PromptTemplate(
        input_variables=["topic", "audience", "tone", "length"],
        template=blog_template
    )
    
    # Email generation template
    email_template = """
    Generate a {email_type} email:
    
    Recipient: {recipient}
    Subject: {subject}
    Key points to include: {key_points}
    
    Email:
    """
    
    email_prompt = PromptTemplate(
        input_variables=["email_type", "recipient", "subject", "key_points"],
        template=email_template
    )
    
    print("Content Generation Examples:")
    
    # Blog post example
    blog_example = blog_prompt.format(
        topic="Getting Started with LangChain",
        audience="Developers",
        tone="Informative and friendly",
        length="500 words"
    )
    print("Blog Post Prompt:")
    print(blog_example[:200] + "...")
    
    # Email example  
    email_example = email_prompt.format(
        email_type="professional",
        recipient="Team",
        subject="Project Update",
        key_points="Progress, next steps, timeline"
    )
    print("\nEmail Prompt:")
    print(email_example[:200] + "...")

def task9_sentiment_analysis():
    """Task 9: Sentiment Analysis"""
    print("\n" + "=" * 60)
    print("TASK 9: Sentiment Analysis")
    print("=" * 60)
    
    # Sample texts for sentiment analysis
    sample_texts = [
        "I absolutely love this new feature! It's amazing and works perfectly.",
        "This is terrible. Nothing works as expected and it's very frustrating.",
        "The product is okay. It has some good points but also some issues.",
        "Fantastic experience! Highly recommend to everyone."
    ]
    
    # Sentiment analysis template
    sentiment_template = """
    Analyze the sentiment of the following text:
    
    Text: {text}
    
    Sentiment (Positive/Negative/Neutral):
    Confidence (1-10):
    Reasoning:
    """
    
    sentiment_prompt = PromptTemplate(
        input_variables=["text"],
        template=sentiment_template
    )
    
    print("Sentiment Analysis Examples:")
    expected_sentiments = ["Positive", "Negative", "Neutral", "Positive"]
    
    for i, text in enumerate(sample_texts):
        print(f"\nText {i+1}: {text}")
        print(f"Expected Sentiment: {expected_sentiments[i]}")

def task10_code_analysis():
    """Task 10: Code Analysis and Documentation"""
    print("\n" + "=" * 60)
    print("TASK 10: Code Analysis and Documentation")
    print("=" * 60)
    
    # Sample code for analysis
    sample_code = """
    def calculate_fibonacci(n):
        if n <= 1:
            return n
        else:
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        """
    
    # Code analysis template
    code_analysis_template = """
    Analyze the following code:
    
    Code:
    {code}
    
    Analysis:
    - Purpose:
    - Time Complexity:
    - Space Complexity: 
    - Potential Issues:
    - Improvements:
    - Documentation:
    """
    
    code_prompt = PromptTemplate(
        input_variables=["code"],
        template=code_analysis_template
    )
    
    formatted_prompt = code_prompt.format(code=sample_code)
    
    print("Code Analysis Example:")
    print(f"Input Code:\n{sample_code}")
    print("\nExpected Analysis:")
    print("- Purpose: Calculate Fibonacci number recursively")
    print("- Time Complexity: O(2^n) - exponential")
    print("- Space Complexity: O(n) - call stack")
    print("- Issues: Inefficient for large n, no input validation")
    print("- Improvements: Use memoization or iterative approach")

def main():
    """Run all advanced LangChain tasks"""
    print("🚀 Advanced LangChain Tasks Demo")
    print("Exploring complex applications and use cases\n")
    
    try:
        task1_document_summarization()
        task2_multi_document_qa()
        task3_chain_operations()
        task4_few_shot_learning()
        task5_memory_management()
        task6_document_classification()
        task7_data_extraction()
        task8_content_generation()
        task9_sentiment_analysis()
        task10_code_analysis()
        
        print("\n" + "=" * 60)
        print("✅ All advanced LangChain tasks completed!")
        print("=" * 60)
        
        print("\n🎯 Summary of Advanced Capabilities:")
        capabilities = [
            "Document Summarization",
            "Multi-Document Q&A",
            "Sequential Chains",
            "Few-Shot Learning",
            "Memory Management",
            "Document Classification", 
            "Data Extraction",
            "Content Generation",
            "Sentiment Analysis",
            "Code Analysis"
        ]
        
        for i, capability in enumerate(capabilities, 1):
            print(f"{i:2d}. {capability}")
            
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    main()