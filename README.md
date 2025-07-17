# RAG Application with Google Gemini and FAISS

A complete Retrieval-Augmented Generation (RAG) application that enables intelligent question-answering over PDF documents using Google's Gemini AI and FAISS vector search.

## Overview

This application creates an interactive chat system that can answer questions about PDF documents by:
1. **Document Processing**: Downloads and extracts text from PDFs stored in Google Cloud Storage
2. **Vector Embedding**: Converts text chunks into searchable vector embeddings using Gemini
3. **Similarity Search**: Uses FAISS for efficient semantic search
4. **Answer Generation**: Provides contextual answers using Gemini's language model
5. **Source Citation**: Shows which document sections were used for each answer

## Features

- 🧠 **Intelligent Q&A**: Ask natural language questions about document content
- 📚 **Source Attribution**: See exactly which document sections support each answer
- ⚡ **Fast Search**: FAISS-powered vector similarity search
- 💾 **Persistent Storage**: Save and reuse vector stores to avoid reprocessing
- 🔒 **Secure**: Environment variable-based configuration
- 🌐 **Cloud Integration**: Direct integration with Google Cloud Storage
- 💬 **Interactive Chat**: User-friendly command-line interface

## Prerequisites

### Required Services
- **Google Cloud Platform Account** with billing enabled
- **Google Cloud Storage** bucket with PDF files
- **Gemini API Access** (Google AI Studio or Vertex AI)

### Required Python Version
- Python 3.8 or higher

## Installation

### 1. Clone or Download the Application
```bash
# Save the rag_app.py file to your local directory
```

### 2. Install Dependencies
```bash
pip install langchain langchain-google-genai faiss-cpu pypdf google-cloud-storage python-dotenv
```

### 3. Google Cloud Authentication

#### Option A: Service Account (Recommended for Production)
```bash
# Create a service account in Google Cloud Console
# Download the JSON key file
# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

#### Option B: User Authentication (Development)
```bash
# Install Google Cloud CLI
# Authenticate with your Google account
gcloud auth application-default login
```

## Configuration

### 1. Create Environment File
Create a `.env` file in the same directory as `rag_app.py`:

```env
# Google AI API Configuration
GOOGLE_API_KEY=your-gemini-api-key-here

# Google Cloud Storage Configuration  
GCS_BUCKET_NAME=your-bucket-name
GCS_BLOB_NAME=your-pdf-file.pdf

# Optional: Custom file paths
LOCAL_PDF_PATH=downloaded_document.pdf
VECTOR_STORE_PATH=vector_store
```

### 2. Get Gemini API Key

#### From Google AI Studio (Recommended):
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the generated key to your `.env` file

#### From Vertex AI:
1. Enable Vertex AI API in Google Cloud Console
2. Use service account authentication
3. Set up Vertex AI credentials

### 3. Prepare Google Cloud Storage

#### Create Bucket:
```bash
# Using gcloud CLI
gsutil mb gs://your-bucket-name

# Or use Google Cloud Console
```

#### Upload PDF Files:
```bash
# Upload your PDF to the bucket
gsutil cp your-document.pdf gs://your-bucket-name/

# Or use Google Cloud Console interface
```

#### Set Bucket Permissions:
```bash
# Grant your service account or user account access
gsutil iam ch serviceAccount:your-service-account@project.iam.gserviceaccount.com:objectViewer gs://your-bucket-name
```

## Usage

### 1. Basic Execution
```bash
python rag_app.py
```

### 2. Expected Flow
```
🚀 Starting RAG Application...
📥 Downloading your-document.pdf from bucket your-bucket...
✅ Downloaded your-document.pdf to downloaded_document.pdf
📄 Processing PDF...
✅ Extracted text from downloaded_document.pdf (150 pages).
✅ Split text into 342 chunks with metadata.
🔄 Creating vector store (this may take a while)...
✅ Vector store created successfully!
✅ Vector store saved to vector_store

🤖 RAG CHAT SYSTEM READY!
============================================================
Ask questions about the document content.
Commands:
  - Type your question normally
  - 'quit' or 'exit' to stop  
  - 'help' for this message
============================================================

💬 Your question: 
```

### 3. Sample Interaction
```
💬 Your question: What is the main topic of this document?

==================================================
🔍 Searching for relevant information...
🤖 Generating answer...

🎯 Answer:
Based on the document content, this appears to be a comprehensive guide about machine learning fundamentals, covering topics such as supervised learning, neural networks, and practical implementation strategies.

📚 Sources (3 chunks used):
1. Chunk 15 from your-document.pdf
   Preview: Machine learning is a subset of artificial intelligence that enables computers to learn...

2. Chunk 23 from your-document.pdf  
   Preview: The main approaches to machine learning include supervised, unsupervised, and reinforcement...

3. Chunk 31 from your-document.pdf
   Preview: This guide will walk you through the fundamental concepts and practical applications...

==================================================
```

## File Structure

```
your-project/
├── rag_app.py              # Main application file
├── .env                    # Environment variables (create this)
├── RAG_App_README.md       # This documentation
├── vector_store/           # Generated vector store (auto-created)
│   ├── index.faiss
│   └── index.pkl
└── requirements.txt        # Dependencies (optional)
```

## Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `GOOGLE_API_KEY` | Yes | Gemini API key from Google AI Studio | `AIzaSyBNvT7Fxa976z_-3uHXXOygnJfjuEO6eho` |
| `GCS_BUCKET_NAME` | Yes | Google Cloud Storage bucket name | `my-documents-bucket` |
| `GCS_BLOB_NAME` | Yes | PDF file name in the bucket | `research-paper.pdf` |
| `LOCAL_PDF_PATH` | No | Local filename for downloaded PDF | `document.pdf` (default) |
| `VECTOR_STORE_PATH` | No | Directory for vector store | `vector_store` (default) |

## Advanced Configuration

### Custom Chunk Settings
Modify the text splitter parameters in the code:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # Larger chunks for more context
    chunk_overlap=300,      # More overlap for better continuity
    length_function=len,
    is_separator_regex=False,
)
```

### Custom Model Settings
Adjust Gemini model parameters:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",     # Use Pro model for better quality
    google_api_key=GEMINI_API_KEY,
    temperature=0.1,            # Lower temperature for more factual responses
    max_tokens=1000            # Limit response length
)
```

### Retrieval Settings
Modify search parameters:

```python
# In query_documents function
relevant_docs = vector_store.similarity_search(question, k=5)  # Get more chunks
```

## Troubleshooting

### Common Issues

#### 1. Authentication Errors
```
Error: 403 Forbidden
```
**Solutions:**
- Verify `GOOGLE_API_KEY` is correct
- Check Google Cloud authentication: `gcloud auth list`
- Ensure billing is enabled on your Google Cloud project

#### 2. Bucket Access Issues
```
Error downloading PDF from GCS: 403 Forbidden
```
**Solutions:**
- Verify bucket name and file name are correct
- Check bucket permissions: `gsutil iam get gs://your-bucket-name`
- Ensure service account has `Storage Object Viewer` role

#### 3. API Quota Exceeded
```
Error: 429 Too Many Requests
```
**Solutions:**
- Wait for quota reset
- Check API usage in Google Cloud Console
- Consider upgrading to higher quota limits

#### 4. Memory Issues with Large PDFs
```
Error: Memory allocation failed
```
**Solutions:**
- Reduce `chunk_size` parameter
- Process smaller PDF files
- Increase system memory

#### 5. Vector Store Loading Fails
```
Error loading vector store: No such file or directory
```
**Solutions:**
- Delete existing vector_store directory and recreate
- Check file permissions
- Ensure FAISS is properly installed

### Debug Mode

Enable verbose logging by modifying the code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints in functions
print(f"DEBUG: Processing {len(documents)} chunks...")
```

## Performance Optimization

### 1. Vector Store Reuse
- The application automatically saves vector stores
- Subsequent runs will load existing stores instead of reprocessing
- Delete `vector_store/` directory to force reprocessing

### 2. Chunk Size Optimization
- **Larger chunks**: More context, slower search
- **Smaller chunks**: Faster search, less context
- Optimal range: 800-1500 characters

### 3. Search Parameters
- **Higher k value**: More comprehensive answers, slower
- **Lower k value**: Faster responses, potentially less comprehensive

## Security Best Practices

### 1. Environment Variables
- Never commit `.env` files to version control
- Use different API keys for development and production
- Regularly rotate API keys

### 2. Access Control
```bash
# Limit bucket access to specific service accounts
gsutil iam ch -d user:old-user@gmail.com:objectViewer gs://your-bucket-name
gsutil iam ch serviceAccount:new-service@project.iam.gserviceaccount.com:objectViewer gs://your-bucket-name
```

### 3. Network Security
- Use VPC for production deployments
- Implement proper firewall rules
- Consider using private Google Access

## Customization Examples

### 1. Multiple Document Support
```python
def process_multiple_pdfs(pdf_list):
    all_documents = []
    for pdf_file in pdf_list:
        # Download and process each PDF
        documents = process_single_pdf(pdf_file)
        all_documents.extend(documents)
    return FAISS.from_documents(all_documents, embeddings_model)
```

### 2. Custom Prompt Templates
```python
CUSTOM_PROMPT = """You are an expert document analyst. Answer the question based on the provided context.

Context: {context}

Question: {question}

Provide a detailed answer with specific references to the source material.

Answer:"""
```

### 3. Response Filtering
```python
def filter_response(answer, confidence_threshold=0.8):
    if "I don't know" in answer or "insufficient information" in answer:
        return "The document doesn't contain enough information to answer this question."
    return answer
```

## Cost Optimization

### Gemini API Costs
- **Text Embedding**: ~$0.0001 per 1K characters
- **Text Generation**: ~$0.002 per 1K characters
- **Optimization**: Cache embeddings, use smaller chunks

### Google Cloud Storage
- **Storage**: ~$0.02 per GB per month
- **Operations**: Minimal cost for occasional downloads
- **Optimization**: Use regional buckets close to compute

## Support and Resources

### Documentation Links
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Google Cloud Storage Documentation](https://cloud.google.com/storage/docs)

### Community Resources
- [LangChain Community](https://github.com/langchain-ai/langchain)
- [Google AI Developer Community](https://developers.googleblog.com/search/label/AI)

### Getting Help
1. Check this README for common solutions
2. Review error messages carefully
3. Verify all environment variables are set correctly
4. Test with a simple, small PDF first
5. Check Google Cloud Console for quota and billing status

## License

This project is provided for educational and demonstration purposes. Please review Google's terms of service for Gemini API usage and ensure compliance with your organization's policies.

---

**Happy document chatting! 🤖📚**
