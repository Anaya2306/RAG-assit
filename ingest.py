# Load environment variables from .env file (your API key)
from dotenv import load_dotenv
load_dotenv()

# Import the PDF folder loader
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Import the text splitter - this chops PDFs into small chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import OpenAI embeddings - converts text chunks into vectors
from langchain_openai import OpenAIEmbeddings

# Import ChromaDB - this is our vector database
from langchain_community.vectorstores import Chroma

# Step 1: Load all PDFs from the /docs folder
print("Loading PDFs...")
loader = PyPDFDirectoryLoader("./docs")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Step 2: Split documents into small chunks of 500 characters
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3: Convert chunks to vectors and save to ChromaDB
print("Creating embeddings and saving to ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

print("Done! Your docs are ready to query.")
```

Before running it, you need two things:

**1.** Create a `.env` file in your project folder with your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

**2.** Drop at least one PDF into your `docs/` folder.

Then in the terminal run:
```
py ingest.py