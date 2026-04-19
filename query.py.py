# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import OpenAI chat model
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Import ChromaDB
from langchain_community.vectorstores import Chroma

# Import prompt tools
from langchain.prompts import ChatPromptTemplate

# Step 1: Connect to your existing ChromaDB (already built by ingest.py)
print("Loading your knowledge base...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

# Step 2: Set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Step 3: The prompt template - tells GPT to only use your docs
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If the answer is not in the context, say "I couldn't find that in your documents."

Context:
{context}

Question:
{question}
""")

# Step 4: Ask questions in a loop
print("Ready! Type your question (or 'quit' to exit)\n")

while True:
    question = input("You: ")
    
    if question.lower() == "quit":
        break
    
    # Find the 4 most relevant chunks from your PDFs
    results = vectorstore.similarity_search(question, k=4)
    
    # Combine the chunks into one block of context
    context = "\n\n".join([r.page_content for r in results])
    
    # Ask the LLM with the context + question
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    # Print the answer
    print(f"\nAssistant: {response.content}")
    
    # Show which files the answer came from
    sources = set([r.metadata.get("source", "unknown") for r in results])
    print(f"Sources: {', '.join(sources)}\n")


