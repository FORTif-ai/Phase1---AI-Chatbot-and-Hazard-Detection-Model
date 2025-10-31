import os
import uuid

# Qdrant-specific imports
from qdrant_client import QdrantClient, models

# LangChain-specific imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# --- 1. Configuration & Initialization ---
print("--- Starting Fortif.ai Data Ingestion Pipeline ---")

# Define constants for configuration.
DATA_PATH = "personal_data"
QDRANT_COLLECTION_NAME = "fortif_ai_memory"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIMENSION = 1536

# Check for the OpenAI API key from environment variables.
# The script will fail if this is not set in your terminal.
if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Initialize the clients.
# The Qdrant client will connect to your local Docker container.
qdrant_client = QdrantClient("http://localhost:6333")
# The OpenAI embeddings client will use the API key from the environment.
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

print("Configuration loaded and clients initialized.")

# --- 2. Define Metadata Strategy ---
# This map is the core of your safety and personalization system.
METADATA_MAP = {
    "routine.txt": {"topic": "daily_routine", "is_sensitive": False, "source_file": "routine.txt"},
    "happy_memory.txt": {"topic": "positive_memory", "is_sensitive": False, "source_file": "happy_memory.txt"},
    "sensitive_topic.txt": {"topic": "family_history", "is_sensitive": True, "source_file": "sensitive_topic.txt"},
}
print("Metadata strategy defined.")

# --- 3. Document Loading and Splitting ---
def load_and_split_documents(path: str) -> list:
    """Loads all .txt files from a directory and splits them into chunks."""
    print(f"Loading documents from '{path}'...")
    loader = DirectoryLoader(
        path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} document(s) and split them into {len(chunked_documents)} chunks.")
    return chunked_documents

# --- 4. Data Ingestion into Qdrant ---
def ingest_data_to_qdrant(documents: list):
    """Generates embeddings and upserts the data into Qdrant."""
    print("Starting data ingestion into Qdrant...")
    
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"Collection '{QDRANT_COLLECTION_NAME}' created or reset.")

    texts_to_embed = [doc.page_content for doc in documents]
    metadata_payloads = [
        METADATA_MAP.get(os.path.basename(doc.metadata.get("source")), {})
        for doc in documents
    ]

    print(f"Generating embeddings for {len(texts_to_embed)} text chunks...")
    vectors = embedding_model.embed_documents(texts_to_embed)
    print("Embeddings generated successfully.")

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, **metadata},
            )
            for text, metadata, vector in zip(texts_to_embed, metadata_payloads, vectors)
        ],
        wait=True,
    )
    print(f"Successfully upserted {len(vectors)} points into Qdrant.")

# --- Main Execution Block ---
# This is the entry point of your script.
if __name__ == "__main__":
    # Create dummy data for demonstration purposes if it doesn't exist.
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    with open(os.path.join(DATA_PATH, "routine.txt"), "w") as f:
        f.write("My morning routine is very important. I wake at 7 AM for medication, then have tea and toast.")
    with open(os.path.join(DATA_PATH, "happy_memory.txt"), "w") as f:
        f.write("I remember my granddaughter's 5th birthday. We gave her a red bicycle at the park.")
    with open(os.path.join(DATA_PATH, "sensitive_topic.txt"), "w") as f:
        f.write("My late husband, John, passed away in the winter of 2018 after a long illness.")

    # Run the main logic.
    docs = load_and_split_documents(DATA_PATH)
    if docs:
        # In your local environment, this line will execute the full process.
        ingest_data_to_qdrant(docs)
    else:
        print("No documents found to ingest.")
        
    print("\n--- Ingestion Pipeline Complete! ---")
    print(f"Your data is now in the '{QDRANT_COLLECTION_NAME}' collection in Qdrant.")