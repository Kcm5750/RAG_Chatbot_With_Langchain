"""
RAG Chatbot - Streamlit Application

A retrieval augmented generation chatbot with document upload, vector store,
and conversational QA using LangChain and LLM providers (OpenAI / Gemini / HuggingFace).

Main entry point for Streamlit Community Cloud deployment.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from pathlib import Path
from typing import Optional

# Import our modular components
from src.config import (
    get_config,
    load_streamlit_secrets,
    LLM_PROVIDERS,
    RETRIEVER_TYPES,
    WELCOME_MESSAGES,
    SUPPORTED_FILE_TYPES
)
from src.data_loader import (
    clear_temp_files,
    save_uploaded_files,
    load_documents,
    split_documents
)
from src.vectorstore import (
    get_embeddings,
    create_vectorstore,
    load_vectorstore,
    create_retriever
)
from src.rag_chain import (
    create_conversational_chain,
    format_source_documents
)


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot - Chat With Your Data",
    page_icon="🤖",
    layout="wide"
)

# Load configuration
config = get_config()
load_streamlit_secrets()

# Get base path for the application
BASE_PATH = Path(__file__).resolve().parent
TMP_DIR = config.get_tmp_dir(BASE_PATH)
VECTOR_STORE_DIR = config.get_vector_store_dir(BASE_PATH)

# Ensure directories exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chain" not in st.session_state:
        st.session_state.chain = None
    
    if "memory" not in st.session_state:
        st.session_state.memory = None
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "vectorstore_loaded" not in st.session_state:
        st.session_state.vectorstore_loaded = False
    
    # LLM settings
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "OpenAI"
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = config.default_openai_model
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = config.default_temperature
    
    if "top_p" not in st.session_state:
        st.session_state.top_p = config.default_top_p
    
    if "assistant_language" not in st.session_state:
        st.session_state.assistant_language = "english"
    
    if "retriever_type" not in st.session_state:
        st.session_state.retriever_type = "Contextual compression"
    
    # API keys (try to load from config/secrets)
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = config.openai_api_key or ""
    
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = config.google_api_key or ""
    
    if "hf_api_key" not in st.session_state:
        st.session_state.hf_api_key = config.huggingface_api_key or ""
    
    if "cohere_api_key" not in st.session_state:
        st.session_state.cohere_api_key = config.cohere_api_key or ""


def render_sidebar():
    """Render the sidebar with LLM provider and model selection."""
    with st.sidebar:
        st.caption(
            "🚀 A retrieval augmented generation chatbot powered by 🔗 LangChain, "
            "OpenAI, Google Generative AI, and 🤗 HuggingFace"
        )
        st.write("")
        
        # Provider selection
        provider_options = list(LLM_PROVIDERS.keys())
        provider_display = [LLM_PROVIDERS[p]["display_name"] for p in provider_options]
        
        provider_captions = []
        for p in provider_options:
            if "pricing_link" in LLM_PROVIDERS[p]:
                provider_captions.append(f"[Pricing]({LLM_PROVIDERS[p]['pricing_link']})")
            elif "rate_limit" in LLM_PROVIDERS[p]:
                provider_captions.append(LLM_PROVIDERS[p]["rate_limit"])
            elif "note" in LLM_PROVIDERS[p]:
                provider_captions.append(LLM_PROVIDERS[p]["note"])
            else:
                provider_captions.append("")
        
        selected_display = st.radio(
            "Select LLM Provider",
            provider_display,
            captions=provider_captions
        )
        
        # Map display back to provider key
        selected_idx = provider_display.index(selected_display)
        st.session_state.llm_provider = provider_options[selected_idx]
        
        st.divider()
        
        # API Key input
        provider_info = LLM_PROVIDERS[st.session_state.llm_provider]
        api_key_link = provider_info.get("api_key_link", "")
        
        if st.session_state.llm_provider == "OpenAI":
            st.session_state.openai_api_key = st.text_input(
                f"OpenAI API Key - [Get an API key]({api_key_link})",
                type="password",
                value=st.session_state.openai_api_key,
                placeholder="sk-..."
            )
        elif st.session_state.llm_provider == "Google":
            st.session_state.google_api_key = st.text_input(
                f"Google API Key - [Get an API key]({api_key_link})",
                type="password",
                value=st.session_state.google_api_key,
                placeholder="AIza..."
            )
        elif st.session_state.llm_provider == "HuggingFace":
            st.session_state.hf_api_key = st.text_input(
                f"HuggingFace API Key - [Get an API key]({api_key_link})",
                type="password",
                value=st.session_state.hf_api_key,
                placeholder="hf_..."
            )
        
        # Model and parameters
        with st.expander("**Models and Parameters**"):
            models = provider_info["models"]
            st.session_state.selected_model = st.selectbox(
                f"Choose {st.session_state.llm_provider} model",
                models,
                index=0
            )
            
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1
            )
            
            st.session_state.top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.top_p,
                step=0.05
            )
        
        # Assistant language
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            "Assistant Language",
            list(WELCOME_MESSAGES.keys())
        )
        
        st.divider()
        
        # Retriever settings
        st.subheader("Retriever Settings")
        
        # Filter retriever types based on model
        available_retrievers = RETRIEVER_TYPES.copy()
        if st.session_state.selected_model == "gpt-3.5-turbo":
            # Remove vectorstore backed retriever for gpt-3.5-turbo (token limit risk)
            available_retrievers = [r for r in available_retrievers if r != "Vectorstore backed retriever"]
        
        st.session_state.retriever_type = st.selectbox(
            "Retriever Type",
            available_retrievers
        )
        
        if st.session_state.retriever_type == "Cohere reranker":
            st.session_state.cohere_api_key = st.text_input(
                "Cohere API Key - [Get an API key](https://dashboard.cohere.com/api-keys)",
                type="password",
                value=st.session_state.cohere_api_key,
                placeholder="..."
            )
        
        st.write("")
        st.info(
            f"ℹ️ Your {st.session_state.llm_provider} API key, model parameters, "
            f"and retriever settings are applied when creating or loading a vectorstore."
        )


def get_current_api_key() -> Optional[str]:
    """Get the API key for the current provider."""
    if st.session_state.llm_provider == "OpenAI":
        return st.session_state.openai_api_key
    elif st.session_state.llm_provider == "Google":
        return st.session_state.google_api_key
    elif st.session_state.llm_provider == "HuggingFace":
        return st.session_state.hf_api_key
    return None


def validate_inputs() -> tuple[bool, str]:
    """
    Validate required inputs.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []
    
    # Check API key
    api_key = get_current_api_key()
    if not api_key:
        errors.append(f"insert your {st.session_state.llm_provider} API key")
    
    # Check Cohere API key if using Cohere reranker
    if st.session_state.retriever_type == "Cohere reranker" and not st.session_state.cohere_api_key:
        errors.append("insert your Cohere API key")
    
    if errors:
        if len(errors) == 1:
            return False, f"Please {errors[0]}."
        else:
            return False, f"Please {', '.join(errors[:-1])}, and {errors[-1]}."
    
    return True, ""


def create_vectorstore_from_uploads():
    """Create a new vectorstore from uploaded documents."""
    uploaded_files = st.session_state.get("uploaded_files", [])
    vectorstore_name = st.session_state.get("vectorstore_name", "").strip()
    
    # Validate inputs
    errors = []
    is_valid, error_msg = validate_inputs()
    if not is_valid:
        errors.append(error_msg)
    
    if not uploaded_files:
        errors.append("Please select documents to upload.")
    
    if not vectorstore_name:
        errors.append("Please provide a vectorstore name.")
    
    if errors:
        for error in errors:
            st.error(error)
        return
    
    try:
        with st.spinner("Processing documents and creating vectorstore..."):
            # Clear temp directory
            clear_temp_files(TMP_DIR)
            
            # Save uploaded files
            save_errors = save_uploaded_files(uploaded_files, TMP_DIR)
            if save_errors:
                st.warning("Some files had errors:\n" + "\n".join(save_errors))
            
            # Load documents
            documents = load_documents(TMP_DIR)
            if not documents:
                st.error("No documents were loaded. Please check your files.")
                return
            
            st.info(f"Loaded {len(documents)} document(s)")
            
            # Split documents
            chunks = split_documents(
                documents,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            st.info(f"Split into {len(chunks)} chunks")
            
            # Get embeddings
            api_key = get_current_api_key()
            embeddings = get_embeddings(
                provider=st.session_state.llm_provider,
                api_key=api_key
            )
            
            # Create vectorstore
            persist_dir = str(VECTOR_STORE_DIR / vectorstore_name)
            st.session_state.vector_store = create_vectorstore(
                chunks=chunks,
                embeddings=embeddings,
                persist_dir=persist_dir
            )
            
            # Create retriever
            st.session_state.retriever = create_retriever(
                vectorstore=st.session_state.vector_store,
                embeddings=embeddings,
                retriever_type=st.session_state.retriever_type,
                base_retriever_search_type="similarity",
                base_retriever_k=config.base_retriever_k,
                compression_retriever_k=config.compression_retriever_k,
                compression_chunk_size=config.compression_chunk_size,
                cohere_api_key=st.session_state.cohere_api_key if st.session_state.cohere_api_key else None,
                cohere_model=config.cohere_model,
                cohere_top_n=config.cohere_top_n
            )
            
            # Create conversational chain
            st.session_state.chain, st.session_state.memory = create_conversational_chain(
                retriever=st.session_state.retriever,
                provider=st.session_state.llm_provider,
                model=st.session_state.selected_model,
                api_key=api_key,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                language=st.session_state.assistant_language,
                memory_max_token=config.memory_max_token
            )
            
            # Clear chat history
            clear_chat_history()
            
            st.session_state.vectorstore_loaded = True
            st.success(f"✅ Vectorstore **{vectorstore_name}** created successfully!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def load_existing_vectorstore(vectorstore_path: str):
    """Load an existing vectorstore."""
    is_valid, error_msg = validate_inputs()
    if not is_valid:
        st.error(error_msg)
        return
    
    try:
        with st.spinner("Loading vectorstore..."):
            # Get embeddings
            api_key = get_current_api_key()
            embeddings = get_embeddings(
                provider=st.session_state.llm_provider,
                api_key=api_key
            )
            
            # Load vectorstore
            st.session_state.vector_store = load_vectorstore(
                embeddings=embeddings,
                persist_dir=vectorstore_path
            )
            
            # Create retriever
            st.session_state.retriever = create_retriever(
                vectorstore=st.session_state.vector_store,
                embeddings=embeddings,
                retriever_type=st.session_state.retriever_type,
                base_retriever_search_type="similarity",
                base_retriever_k=config.base_retriever_k,
                compression_retriever_k=config.compression_retriever_k,
                compression_chunk_size=config.compression_chunk_size,
                cohere_api_key=st.session_state.cohere_api_key if st.session_state.cohere_api_key else None,
                cohere_model=config.cohere_model,
                cohere_top_n=config.cohere_top_n
            )
            
            # Create conversational chain
            st.session_state.chain, st.session_state.memory = create_conversational_chain(
                retriever=st.session_state.retriever,
                provider=st.session_state.llm_provider,
                model=st.session_state.selected_model,
                api_key=api_key,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                language=st.session_state.assistant_language,
                memory_max_token=config.memory_max_token
            )
            
            # Clear chat history
            clear_chat_history()
            
            st.session_state.vectorstore_loaded = True
            vectorstore_name = Path(vectorstore_path).name
            st.success(f"✅ Vectorstore **{vectorstore_name}** loaded successfully!")
            
    except Exception as e:
        st.error(f"An error occurred while loading vectorstore: {str(e)}")


def render_vectorstore_tabs():
    """Render tabs for creating or loading vectorstores."""
    tab_create, tab_load = st.tabs(["📤 Create New Vectorstore", "📂 Load Existing Vectorstore"])
    
    with tab_create:
        st.markdown("### Upload Documents")
        st.session_state.uploaded_files = st.file_uploader(
            "Select documents to upload",
            accept_multiple_files=True,
            type=SUPPORTED_FILE_TYPES,
            help=f"Supported formats: {', '.join(SUPPORTED_FILE_TYPES).upper()}"
        )
        
        st.session_state.vectorstore_name = st.text_input(
            "Vectorstore Name",
            placeholder="my_documents",
            help="Enter a unique name for this vectorstore"
        )
        
        st.button(
            "🚀 Create Vectorstore",
            on_click=create_vectorstore_from_uploads,
            type="primary",
            use_container_width=True
        )
    
    with tab_load:
        st.markdown("### Load Existing Vectorstore")
        
        # List available vectorstores
        if VECTOR_STORE_DIR.exists():
            vectorstores = [d for d in VECTOR_STORE_DIR.iterdir() if d.is_dir()]
            if vectorstores:
                vectorstore_names = [d.name for d in vectorstores]
                selected_vectorstore = st.selectbox(
                    "Select a vectorstore",
                    vectorstore_names
                )
                
                if st.button("📥 Load Vectorstore", type="primary", use_container_width=True):
                    vectorstore_path = str(VECTOR_STORE_DIR / selected_vectorstore)
                    load_existing_vectorstore(vectorstore_path)
            else:
                st.info("No vectorstores found. Create one using the 'Create New Vectorstore' tab.")
        else:
            st.info("No vectorstores directory found. Create your first vectorstore!")


def clear_chat_history():
    """Clear chat history and memory."""
    st.session_state.messages = [{
        "role": "assistant",
        "content": WELCOME_MESSAGES[st.session_state.assistant_language]
    }]
    
    if st.session_state.memory:
        try:
            st.session_state.memory.clear()
        except Exception:
            pass


def process_chat_input(prompt: str):
    """Process user chat input and generate response."""
    if not st.session_state.vectorstore_loaded or not st.session_state.chain:
        st.error("Please create or load a vectorstore first!")
        return
    
    api_key = get_current_api_key()
    if not api_key:
        st.error(f"Please insert your {st.session_state.llm_provider} API key to continue.")
        return
    
    try:
        with st.spinner("Thinking..."):
            # Invoke the chain
            response = st.session_state.chain.invoke({"question": prompt})
            answer = response["answer"]
            
            # For HuggingFace, extract the answer
            if st.session_state.llm_provider == "HuggingFace":
                if "\nAnswer: " in answer:
                    answer = answer[answer.find("\nAnswer: ") + len("\nAnswer: "):]
            
            # Add to message history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Display source documents
                if response.get("source_documents"):
                    with st.expander("📚 **Source Documents**"):
                        formatted_sources = format_source_documents(response["source_documents"])
                        st.markdown(formatted_sources)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def render_chat_interface():
    """Render the chat interface."""
    st.divider()
    
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("💬 Chat with Your Data")
    with col2:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            clear_chat_history()
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        process_chat_input(prompt)


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Title
    st.title("🤖 RAG Chatbot")
    st.markdown("*Chat with your documents using AI-powered retrieval augmented generation*")
    
    # Render sidebar
    render_sidebar()
    
    # Render vectorstore tabs
    render_vectorstore_tabs()
    
    # Render chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()
