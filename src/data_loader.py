"""
Document loading and chunking functionality for RAG Chatbot.

Handles file uploads, document loading for multiple formats (PDF, TXT, CSV, DOCX),
and text splitting for vector store ingestion.
"""

import os
import glob
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def clear_temp_files(tmp_dir: Path) -> None:
    """
    Delete all files from the temporary directory.
    
    Args:
        tmp_dir: Path to temporary directory
    """
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return
    
    files = glob.glob(str(tmp_dir / "*"))
    for f in files:
        try:
            os.remove(f)
        except Exception:
            pass  # Silently ignore errors


def save_uploaded_files(uploaded_files: List, tmp_dir: Path) -> List[str]:
    """
    Save uploaded files to temporary directory.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        tmp_dir: Path to temporary directory
        
    Returns:
        List of error messages (empty if all successful)
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    
    for uploaded_file in uploaded_files:
        try:
            temp_file_path = tmp_dir / uploaded_file.name
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
        except Exception as e:
            errors.append(f"Error saving {uploaded_file.name}: {str(e)}")
    
    return errors


def load_documents(tmp_dir: Path) -> List[Document]:
    """
    Load documents from temporary directory using LangChain loaders.
    Supports PDF, TXT, CSV, and DOCX formats.
    
    Args:
        tmp_dir: Path to directory containing documents
        
    Returns:
        List of loaded Document objects
    """
    documents = []
    tmp_dir_str = str(tmp_dir)
    
    # Load TXT files
    try:
        txt_loader = DirectoryLoader(
            tmp_dir_str,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents.extend(txt_loader.load())
    except Exception:
        pass  # No txt files or error loading
    
    # Load PDF files
    try:
        pdf_loader = DirectoryLoader(
            tmp_dir_str,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents.extend(pdf_loader.load())
    except Exception:
        pass  # No pdf files or error loading
    
    # Load CSV files
    try:
        csv_loader = DirectoryLoader(
            tmp_dir_str,
            glob="**/*.csv",
            loader_cls=CSVLoader,
            show_progress=True,
            loader_kwargs={"encoding": "utf8"}
        )
        documents.extend(csv_loader.load())
    except Exception:
        pass  # No csv files or error loading
    
    # Load DOCX files
    try:
        doc_loader = DirectoryLoader(
            tmp_dir_str,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True
        )
        documents.extend(doc_loader.load())
    except Exception:
        pass  # No docx files or error loading
    
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1600,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_document_count(tmp_dir: Path) -> dict:
    """
    Count documents by type in the temporary directory.
    
    Args:
        tmp_dir: Path to temporary directory
        
    Returns:
        Dictionary with counts by file type
    """
    if not tmp_dir.exists():
        return {"pdf": 0, "txt": 0, "csv": 0, "docx": 0, "total": 0}
    
    counts = {
        "pdf": len(list(tmp_dir.glob("*.pdf"))),
        "txt": len(list(tmp_dir.glob("*.txt"))),
        "csv": len(list(tmp_dir.glob("*.csv"))),
        "docx": len(list(tmp_dir.glob("*.docx")))
    }
    counts["total"] = sum(counts.values())
    
    return counts
