# rag_app.py
import os
from dotenv import load_dotenv # Import for loading .env file
from google.cloud import storage
from pypdf import PdfReader

# Ensure you have installed langchain, langchain-google-genai, faiss-cpu
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

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
GCS_BLOB_NAME = "harrypotter.pdf" # Make sure this matches your uploaded file name

# --- Step 1: Initialize Gemini Models ---
# The google_api_key can be passed directly or picked up from GOOGLE_API_KEY env var
# For ChatGoogleGenerativeAI (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# For GoogleGenerativeAIEmbeddings (Embedding Model)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
print("Gemini LLM and Embeddings models initialized.")

# --- Step 2: Data Ingestion & Preprocessing (from GCS) ---
def download_pdf_from_gcs(bucket_name, blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {blob_name} to {destination_file_name}")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(f"Extracted text from {pdf_path}.")
    return text

def chunk_text(text):
    """Splits text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    print(f"Split text into {len(chunks)} chunks.")
    return chunks

# --- Main Execution Flow ---
if __name__ == "__main__":
    print(f"Attempting to download {GCS_BLOB_NAME} from bucket {GCS_BUCKET_NAME}...")
    try:
        download_pdf_from_gcs(GCS_BUCKET_NAME, GCS_BLOB_NAME, LOCAL_PDF_PATH)
    except Exception as e:
        print(f"Error downloading PDF from GCS: {e}")
        print("Please ensure:")
        print(f"  - Your bucket '{GCS_BUCKET_NAME}' exists and contains '{GCS_BLOB_NAME}'.")
        print("  - Your Google Cloud project has the 'Storage Object Viewer' role for the service account/user running this.")
        print("  - Your local environment is authenticated to Google Cloud (e.g., via `gcloud auth application-default login`).")
        exit()

    raw_text = extract_text_from_pdf(LOCAL_PDF_PATH)
    chunks = chunk_text(raw_text)

    # --- Step 3: Embedding Generation & Vector Store ---
    print("Generating embeddings and building vector store. This may take a while...")
    try:
        vector_store = FAISS.from_texts(chunks, embeddings_model)
        print("Vector store created successfully.")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("Please ensure:")
        print("  - Your GOOGLE_API_KEY is correct and has access to Gemini Embeddings API.")
        print("  - You have sufficient quota for embedding calls.")
        exit()

