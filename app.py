# rag_app.py - Complete RAG Application
import os
from dotenv import load_dotenv
from google.cloud import storage
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# --- Configuration ---
# Load environment variables from .env file 
load_dotenv()

# Get API Key and GCS Bucket Name from environment variables
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")



if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file or environment variables.")
if not GCS_BUCKET_NAME:
    raise ValueError("GCS_BUCKET_NAME not found in .env file or environment variables.")

# Path for the downloaded PDF
LOCAL_PDF_PATH = "harrypotter.pdf"
GCS_BLOB_NAME = "harrypotter.pdf" 

# --- Step 1: Initialize Gemini Models ---
print("Initializing Gemini models...")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=GEMINI_API_KEY,
    temperature=0.3  # Slightly creative but mostly factual
)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=GEMINI_API_KEY
)
print("✅ Gemini LLM and Embeddings models initialized.")

# --- Step 2: Data Ingestion & Preprocessing Functions ---
def download_pdf_from_gcs(bucket_name, blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"✅ Downloaded {blob_name} to {destination_file_name}")
        return True
    except Exception as e:
        print(f"❌ Error downloading PDF from GCS: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        print(f"✅ Extracted text from {pdf_path} ({len(reader.pages)} pages).")
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return None

def chunk_text_with_metadata(text, source_file):
    """Splits text into smaller, manageable chunks with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split text into chunks
    text_chunks = text_splitter.split_text(text)
    
    # Create Document objects with metadata
    documents = []
    for i, chunk in enumerate(text_chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source_file,
                "chunk_id": i,
                "total_chunks": len(text_chunks)
            }
        )
        documents.append(doc)
    
    print(f"✅ Split text into {len(documents)} chunks with metadata.")
    return documents

# --- Step 3: RAG Query Functions ---
def query_documents(question, vector_store, llm, k=3):
    """Query the document knowledge base and return answer with sources."""
    try:
        # Step 1: Retrieve relevant chunks using similarity search
        print(f"🔍 Searching for relevant information...")
        relevant_docs = vector_store.similarity_search(question, k=k)
        
        if not relevant_docs:
            return "I couldn't find relevant information to answer your question.", []
        
        # Step 2: Create context from retrieved chunks
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            chunk_info = f"[Chunk {i+1}]"
            context_parts.append(f"{chunk_info}\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Create enhanced prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so
- Be specific and cite relevant parts when possible
- Keep your answer clear and concise

Answer:"""
        
        # Step 4: Generate response
        print(f"🤖 Generating answer...")
        response = llm.invoke(prompt)
        
        return response.content, relevant_docs
        
    except Exception as e:
        error_msg = f"Error processing question: {e}"
        print(f"❌ {error_msg}")
        return error_msg, []

def display_sources(docs):
    """Display information about the source documents."""
    if not docs:
        return
    
    print(f"\n📚 Sources ({len(docs)} chunks used):")
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"{i+1}. Chunk {metadata.get('chunk_id', 'N/A')} from {metadata.get('source', 'Unknown')}")
        print(f"   Preview: {preview}...")
        print()

def chat_with_documents(vector_store, llm):
    """Interactive chat interface for the RAG system."""
    print("\n" + "="*60)
    print("🤖 RAG CHAT SYSTEM READY!")
    print("="*60)
    print("Ask questions about the document content.")
    print("Commands:")
    print("  - Type your question normally")
    print("  - 'quit' or 'exit' to stop")
    print("  - 'help' for this message")
    print("="*60)
    
    while True:
        try:
            question = input("\n💬 Your question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for using the RAG system.")
                break
                
            if question.lower() == 'help':
                print("\n📖 How to use:")
                print("- Ask any question about the document content")
                print("- Example: 'What is the main character's name?'")
                print("- Example: 'Summarize the first chapter'")
                continue
            
            print(f"\n{'='*50}")
            
            # Get answer and sources
            answer, sources = query_documents(question, vector_store, llm, k=3)
            
            # Display answer
            print(f"\n🎯 Answer:")
            print(answer)
            
            # Display sources
            if sources:
                display_sources(sources)
            
            print(f"{'='*50}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try asking your question again.")

# --- Step 4: Save and Load Vector Store ---
def save_vector_store(vector_store, path="vector_store"):
    """Save the vector store to disk for reuse."""
    try:
        vector_store.save_local(path)
        print(f"✅ Vector store saved to {path}")
    except Exception as e:
        print(f"❌ Error saving vector store: {e}")

def load_vector_store(path="vector_store", embeddings_model=None):
    """Load a previously saved vector store."""
    try:
        vector_store = FAISS.load_local(path, embeddings_model)
        print(f"✅ Vector store loaded from {path}")
        return vector_store
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None

# --- Main Execution Flow ---
def main():
    print("🚀 Starting RAG Application...")
    
    # Check if we can load existing vector store
    existing_store = load_vector_store("vector_store", embeddings_model)
    if existing_store:
        print("📁 Using existing vector store.")
        chat_with_documents(existing_store, llm)
        return
    
    # Step 1: Download PDF from GCS
    print(f"\n📥 Downloading {GCS_BLOB_NAME} from bucket {GCS_BUCKET_NAME}...")
    
    if not download_pdf_from_gcs(GCS_BUCKET_NAME, GCS_BLOB_NAME, LOCAL_PDF_PATH):
        print("\n❌ Failed to download PDF. Please check:")
        print(f"  - Bucket '{GCS_BUCKET_NAME}' exists and contains '{GCS_BLOB_NAME}'")
        print("  - Google Cloud authentication is set up")
        print("  - You have necessary permissions")
        return
    
    # Step 2: Extract and process text
    print(f"\n📄 Processing PDF...")
    raw_text = extract_text_from_pdf(LOCAL_PDF_PATH)
    if not raw_text:
        print("❌ Failed to extract text from PDF.")
        return
    
    # Step 3: Create chunks with metadata
    documents = chunk_text_with_metadata(raw_text, LOCAL_PDF_PATH)
    if not documents:
        print("❌ Failed to create document chunks.")
        return
    
    # Step 4: Create vector store
    print(f"\n🔄 Creating vector store (this may take a while)...")
    try:
        vector_store = FAISS.from_documents(documents, embeddings_model)
        print("✅ Vector store created successfully!")
        
        # Save for future use
        save_vector_store(vector_store, "vector_store")
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        print("Please check:")
        print("  - Your GOOGLE_API_KEY is valid")
        print("  - You have sufficient API quota")
        print("  - Internet connection is stable")
        return
    
    # Step 5: Start interactive chat
    chat_with_documents(vector_store, llm)
    
    # Cleanup
    try:
        if os.path.exists(LOCAL_PDF_PATH):
            os.remove(LOCAL_PDF_PATH)
            print(f"🧹 Cleaned up temporary file: {LOCAL_PDF_PATH}")
    except Exception as e:
        print(f"Warning: Could not remove temporary file: {e}")

if __name__ == "__main__":
    main()
