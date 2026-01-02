"""
Vector store and embeddings management for RAG Chatbot.

Handles embeddings model selection, vector store creation/loading,
and retriever configuration (base, compression, Cohere reranker).
"""

from pathlib import Path
from typing import List, Optional, Union

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    CohereRerank
)


def get_embeddings(
    provider: str,
    api_key: str,
    model_name: Optional[str] = None
):
    """
    Get embeddings model based on provider.
    
    Args:
        provider: LLM provider name ("OpenAI", "Google", "HuggingFace")
        api_key: API key for the provider
        model_name: Optional specific model name
        
    Returns:
        Embeddings model instance
    """
    if provider == "OpenAI":
        return OpenAIEmbeddings(api_key=api_key)
    
    elif provider == "Google":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
    
    elif provider == "HuggingFace":
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key,
            model_name=model_name or "thenlper/gte-large"
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_vectorstore(
    chunks: List[Document],
    embeddings,
    persist_dir: str
) -> Chroma:
    """
    Create a new Chroma vector store from document chunks.
    
    Args:
        chunks: List of document chunks
        embeddings: Embeddings model instance
        persist_dir: Directory to persist the vector store
        
    Returns:
        Chroma vector store instance
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


def load_vectorstore(
    embeddings,
    persist_dir: str
) -> Chroma:
    """
    Load an existing Chroma vector store.
    
    Args:
        embeddings: Embeddings model instance (must match the one used to create)
        persist_dir: Directory where vector store is persisted
        
    Returns:
        Chroma vector store instance
    """
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


def create_base_retriever(
    vectorstore: Chroma,
    search_type: str = "similarity",
    k: int = 4,
    score_threshold: Optional[float] = None
):
    """
    Create a basic vectorstore-backed retriever.
    
    Args:
        vectorstore: Chroma vector store instance
        search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
        k: Number of documents to return
        score_threshold: Minimum relevance threshold (for similarity_score_threshold)
        
    Returns:
        Vectorstore-backed retriever
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever


def create_compression_retriever(
    embeddings,
    base_retriever,
    chunk_size: int = 500,
    k: int = 16,
    similarity_threshold: Optional[float] = None
):
    """
    Create a contextual compression retriever.
    
    Wraps the base retriever with a compression pipeline that:
    1. Splits documents into smaller chunks
    2. Removes redundant documents
    3. Filters top k relevant documents
    4. Reorders documents (most relevant at beginning/end)
    
    Args:
        embeddings: Embeddings model instance
        base_retriever: Base vectorstore-backed retriever
        chunk_size: Size for splitting documents
        k: Number of top relevant documents to keep
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Contextual compression retriever
    """
    # 1. Split documents into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separator=". "
    )
    
    # 2. Remove redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    
    # 3. Filter based on relevance
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings,
        k=k,
        similarity_threshold=similarity_threshold
    )
    
    # 4. Reorder documents (most relevant at beginning/end)
    reordering = LongContextReorder()
    
    # Create compression pipeline
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever


def create_cohere_retriever(
    base_retriever,
    cohere_api_key: str,
    cohere_model: str = "rerank-multilingual-v2.0",
    top_n: int = 10
):
    """
    Create a Cohere reranking retriever.
    
    Uses Cohere's rerank endpoint to reorder results based on relevance.
    
    Args:
        base_retriever: Base vectorstore-backed retriever
        cohere_api_key: Cohere API key
        cohere_model: Cohere model name
        top_n: Number of top results to return
        
    Returns:
        Cohere reranking retriever
    """
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key,
        model=cohere_model,
        top_n=top_n
    )
    
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return retriever


def create_retriever(
    vectorstore: Chroma,
    embeddings,
    retriever_type: str = "Contextual compression",
    base_retriever_search_type: str = "similarity",
    base_retriever_k: int = 16,
    compression_retriever_k: int = 20,
    compression_chunk_size: int = 500,
    cohere_api_key: Optional[str] = None,
    cohere_model: str = "rerank-multilingual-v2.0",
    cohere_top_n: int = 10
):
    """
    Create a retriever based on the specified type.
    
    Args:
        vectorstore: Chroma vector store instance
        embeddings: Embeddings model instance
        retriever_type: Type of retriever to create
        base_retriever_search_type: Search type for base retriever
        base_retriever_k: Number of documents for base retriever
        compression_retriever_k: Number of documents for compression retriever
        compression_chunk_size: Chunk size for compression retriever
        cohere_api_key: Cohere API key (required for Cohere reranker)
        cohere_model: Cohere model name
        cohere_top_n: Number of top results for Cohere reranker
        
    Returns:
        Configured retriever instance
    """
    # Create base retriever
    base_retriever = create_base_retriever(
        vectorstore=vectorstore,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None
    )
    
    if retriever_type == "Vectorstore backed retriever":
        return base_retriever
    
    elif retriever_type == "Contextual compression":
        return create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            chunk_size=compression_chunk_size,
            k=compression_retriever_k,
            similarity_threshold=None
        )
    
    elif retriever_type == "Cohere reranker":
        if not cohere_api_key:
            raise ValueError("Cohere API key is required for Cohere reranker")
        return create_cohere_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            cohere_model=cohere_model,
            top_n=cohere_top_n
        )
    
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")
