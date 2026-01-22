from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader


class BaseAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, input_data):
        raise NotImplementedError


class LoaderAgent(BaseAgent):
    def run(self, text):
        print("\n[LoaderAgent] Cleaning text...")
        loader = TextLoader(text)
        documents = loader.load()
        return documents[0].page_content


class AnalyzerAgent(BaseAgent):
    def run(self, text):
        print("\n[AnalyzerAgent] Extracting key points...")
        prompt = f"""
        Extract important key points as bullet points.

        TEXT:
        {text}
        """
        response = self.llm.invoke(prompt)
        return response


class SummarizerAgent(BaseAgent):
    def run(self, key_points):
        print("\n[SummarizerAgent] Creating summary...")
        prompt = f"""
        Create a concise summary using these key points:

        {key_points}
        """
        response = self.llm.invoke(prompt)
        return response


class ReviewerAgent(BaseAgent):
    def run(self, summary):
        print("\n[ReviewerAgent] Refining summary...")
        prompt = f"""
        Improve clarity, grammar, and conciseness:

        {summary}
        """
        response = self.llm.invoke(prompt)
        return response


class MultiAgentPipeline:
    def __init__(self, agents):
        self.agents = agents

    def run(self, input_data):
        data = input_data
        for agent in self.agents:
            data = agent.run(data)
        return data


def main():
    llm = Ollama(
        model="llama2",
        temperature=0
    )

    pipeline = MultiAgentPipeline([
        LoaderAgent(llm),
        AnalyzerAgent(llm),
        SummarizerAgent(llm),
        ReviewerAgent(llm)
    ])

    text = 'data.txt'   # for text document

    # # for pdfs only
    # loader = PyPDFLoader("sample.pdf")
    # docs = loader.load()
    # text = " ".join([doc.page_content for doc in docs])

    output = pipeline.run(text)

    print("\n======= FINAL OUTPUT =======\n")
    print(output)


if __name__ == "__main__":
    main()




# ======= FINAL OUTPUT =======

# Here is a revised version of the passage that improves clarity, grammar, and conciseness:

# LangChain is an open-source toolkit for building conversational AI applications and question answering systems. It comprises several modular components: Document Loaders, Text Splitters, Embeddings, Vector Stores, Chains, Memory, and Agents. These components work together to efficiently load documents from various sources, break them down into manageable chunks, convert text into vector representations for similarity search, store and retrieve document embeddings, combine multiple components to create complex workflows, maintain conversation context across interactions, and build autonomous systems that can use tools and make decisions.

# Common use cases for LangChain include building question answering systems, creating chatbots, summarizing large documents, generating content based on existing documents, and assisting users in finding information from large document collections. LangChain supports various LLM providers such as OpenAI, Anthropic, Hugging Face, and local models, and integrates with vector databases like Pinecone, Weaviate, and FAISS for efficient document retrieval.

# Here are the changes I made:

# 1. Improved sentence structure: The original passage has long, complex sentences that can be difficult to follow. I broke up some of these sentences into shorter, simpler ones to improve clarity.
# 2. Corrected grammar and spelling errors: There were several grammatical errors and typos in the original passage. I corrected them to improve readability.
# 3. Simplified language: Some of the language used in the original passage was technical and complex. I simplified it where possible to make it easier for non-experts to understand.      
# 4. Added clarifying phrases: In some places, I added phrases to help clarify the meaning of the text. For example, I added "comprises" to clarify that the toolkit is made up of several components.
# 5. Removed unnecessary words and phrases: I removed any words or phrases that were not essential to the meaning of the passage. This helped to make it more concise and easier to read.  