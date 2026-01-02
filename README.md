# RAG Chatbot powered by 🔗 LangChain, OpenAI, Google Generative AI and HuggingFace 🤗

<div align="center">
  <img src="https://github.com/AlaGrine/RAG_chatabot_with_Langchain/blob/main/data/docs/RAG_architecture.png" >
  <figcaption>RAG architecture with LangChain components.</figcaption>
</div>

## 📋 Project Overview

Although Large Language Models (LLMs) are powerful and capable of generating creative content, they can produce outdated or incorrect information as they are trained on static data. To overcome this limitation, Retrieval Augmented Generation (RAG) systems can be used to connect the LLM to external data and obtain more reliable answers.

This project provides a production-ready RAG chatbot built with [LangChain](https://python.langchain.com/), powered by [OpenAI](https://platform.openai.com/overview), [Google Generative AI](https://ai.google.dev/?hl=en), and [HuggingFace](https://huggingface.co/) APIs. You can upload documents in **PDF, TXT, CSV, or DOCX** formats and chat with your data. Relevant documents are retrieved and sent to the LLM along with your questions for accurate, context-aware answers.

### ✨ Features

- 🔄 **Multiple LLM Providers**: OpenAI (GPT-3.5, GPT-4), Google (Gemini), HuggingFace (Mistral)
- 📄 **Multi-format Support**: PDF, TXT, CSV, DOCX document loading
- 🗂️ **Vector Store Management**: Create and load Chroma vector databases
- 🔍 **Advanced Retrieval**: Contextual compression, Cohere reranking, similarity search
- 💬 **Conversational Memory**: Maintains context across chat sessions
- 🌍 **Multi-language Support**: Responses in 10+ languages
- 🎨 **Clean UI**: Modern Streamlit interface with source document display
- ☁️ **Cloud-Ready**: Deployable to Streamlit Community Cloud

## 🏗️ Project Structure

```
RAG_chatabot_with_Langchain-main/
├── src/                          # Modular source package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── data_loader.py           # Document loading and chunking
│   ├── vectorstore.py           # Embeddings and vector store
│   └── rag_chain.py             # RAG chain and memory
├── data/
│   ├── tmp/                     # Temporary uploaded files
│   └── vector_stores/           # Persisted vector databases
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── streamlit_app.py             # Main application entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG_chatabot_with_Langchain-main
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your API keys
   # You can use any text editor
   ```

   Add your API keys to `.env`:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   GOOGLE_API_KEY=your-google-key-here
   HUGGINGFACE_API_KEY=hf_your-key-here
   COHERE_API_KEY=your-cohere-key-here  # Optional
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

   The app will open in your browser at `http://localhost:8501`

## ☁️ Deployment to Streamlit Community Cloud

### Step 1: Prepare Your Repository

1. Push your code to GitHub (ensure `.env` is in `.gitignore`)
2. Make sure `streamlit_app.py` is in the root directory
3. Verify `requirements.txt` is up to date

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch, and `streamlit_app.py`
5. Click "Deploy"

### Step 3: Configure Secrets

In the Streamlit Cloud dashboard:

1. Go to your app settings
2. Click on "Secrets" in the left sidebar
3. Add your API keys in TOML format:

```toml
OPENAI_API_KEY = "sk-your-key-here"
GOOGLE_API_KEY = "your-google-key-here"
HUGGINGFACE_API_KEY = "hf_your-key-here"
COHERE_API_KEY = "your-cohere-key-here"
```

4. Click "Save"
5. Your app will automatically restart with the new secrets

### Important Notes for Cloud Deployment

- **API Keys**: Never commit API keys to your repository. Always use Streamlit secrets.
- **File Uploads**: Streamlit Cloud has a default 200MB upload limit (configurable in `.streamlit/config.toml`)
- **Vector Stores**: Vector stores are persisted in the app's ephemeral storage. They will be lost when the app restarts. For production, consider using a persistent storage solution.
- **Memory**: Free tier has limited memory. Large documents may require optimization.

## 📖 Usage Instructions

### 1. Select Your LLM Provider

In the sidebar:
- Choose between OpenAI, Google Generative AI, or HuggingFace
- Enter your API key for the selected provider
- Select a model and adjust parameters (temperature, top_p)

### 2. Create a Vector Store

**Option A: Upload New Documents**
1. Go to the "Create New Vectorstore" tab
2. Upload one or more documents (PDF, TXT, CSV, DOCX)
3. Enter a unique name for your vectorstore
4. Click "Create Vectorstore"
5. Wait for processing to complete

**Option B: Load Existing Vectorstore**
1. Go to the "Load Existing Vectorstore" tab
2. Select a previously created vectorstore from the dropdown
3. Click "Load Vectorstore"

### 3. Chat with Your Documents

1. Once a vectorstore is loaded, use the chat input at the bottom
2. Ask questions about your documents
3. View answers with source document citations
4. Click on "Source Documents" to see the retrieved context

### 4. Advanced Settings

- **Retriever Type**: Choose between:
  - **Contextual Compression**: Filters and reorders documents for relevance
  - **Cohere Reranker**: Uses Cohere API to rerank results (requires Cohere API key)
  - **Vectorstore Backed Retriever**: Basic similarity search
  
- **Assistant Language**: Select the language for AI responses

## 🔑 Getting API Keys

- **OpenAI**: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
- **Google AI**: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- **HuggingFace**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **Cohere** (optional): [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)

## 🛠️ Configuration

### Environment Variables

All configuration can be customized via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `GOOGLE_API_KEY` | Google AI API key | None |
| `HUGGINGFACE_API_KEY` | HuggingFace API key | None |
| `COHERE_API_KEY` | Cohere API key | None |
| `CHUNK_SIZE` | Document chunk size | 1600 |
| `CHUNK_OVERLAP` | Chunk overlap | 200 |
| `DEFAULT_TEMPERATURE` | LLM temperature | 0.5 |
| `DEFAULT_TOP_P` | LLM top_p | 0.95 |

### Streamlit Configuration

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Upload size limits
- Server settings

## 🐛 Troubleshooting

### Common Issues

**"Please insert your API key"**
- Ensure you've entered a valid API key for your selected provider
- Check that the key is correctly set in `.env` or Streamlit secrets

**"No documents were loaded"**
- Verify your files are in supported formats (PDF, TXT, CSV, DOCX)
- Check file encoding (use UTF-8 for text files)
- Ensure files are not corrupted

**"Token limit exceeded"**
- Use smaller documents or increase chunking
- For GPT-3.5, avoid "Vectorstore backed retriever" (use compression instead)
- Reduce the number of retrieved documents

**Vector store not persisting on Streamlit Cloud**
- This is expected behavior. Streamlit Cloud uses ephemeral storage
- Vector stores are recreated when the app restarts
- For production, integrate with a persistent vector database service

## 📚 Learn More

- **Blog Post**: [RAG Chatbot with LangChain](https://medium.com/@alaeddine.grine/rag-chatbot-powered-by-langchain-openai-google-generative-ai-and-hugging-face-apis-6a9b9d7d59db)
- **LangChain Docs**: [https://python.langchain.com/](https://python.langchain.com/)
- **Streamlit Docs**: [https://docs.streamlit.io/](https://docs.streamlit.io/)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- UI powered by [Streamlit](https://streamlit.io/)
- Vector storage by [Chroma](https://www.trychroma.com/)
- LLM providers: OpenAI, Google, HuggingFace, Cohere

---

**Note**: The original `RAG_app.py` is kept for reference but has been superseded by the modular `streamlit_app.py` architecture.
