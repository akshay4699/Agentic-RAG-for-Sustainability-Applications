"""
VectorStore service — handles document loading, splitting, embedding, and retrieval.

Uses Pinecone vector store with NVIDIA embeddings for vector search.
Loads and indexes PDF files from the configured data directory.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singletons
_vectorstore: PineconeVectorStore | None = None
_retriever: VectorStoreRetriever | None = None


def _get_embeddings() -> NVIDIAEmbeddings:
    """Return the configured embeddings model."""
    settings = get_settings()
    return NVIDIAEmbeddings(model=settings.EMBEDDING_MODEL)


def load_and_index_pdfs(data_dir: str | Path | None = None) -> int:
    """
    Load PDF documents from data directory, split them, and index into Pinecone.

    Args:
        data_dir: Path to directory containing PDFs. Defaults to settings.DATA_DIR.

    Returns:
        Number of chunks indexed.
    """
    global _vectorstore, _retriever

    settings = get_settings()
    target_dir = Path(data_dir or settings.DATA_DIR).resolve()

    if not target_dir.exists():
        logger.error(f"Data directory does not exist: {target_dir}")
        return 0

    logger.info(f"Loading PDFs from directory: {target_dir}")
    try:
        loader = PyPDFDirectoryLoader(str(target_dir))
        docs = loader.load()
    except Exception as e:
        logger.error(f"Failed to load PDFs from {target_dir}: {e}")
        return 0

    if not docs:
        logger.error(f"No PDF documents found in {target_dir}!")
        return 0

    logger.info(f"Loaded {len(docs)} PDF pages total")

    # --- Split into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"Split PDFs into {len(splits)} chunks")

    # --- Index into Pinecone ---
    try:
        _vectorstore = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=_get_embeddings(),
            index_name=settings.PINECONE_INDEX_NAME,
        )
        _retriever = _vectorstore.as_retriever(
            search_kwargs={"k": settings.RETRIEVAL_K}
        )
        logger.info(
            f"Successfully indexed {len(splits)} chunks into Pinecone index: "
            f"'{settings.PINECONE_INDEX_NAME}'"
        )
        return len(splits)
    except Exception as e:
        logger.error(f"Failed to index documents into Pinecone: {e}")
        raise


def load_and_index_documents(urls: list[str] | None = None) -> int:
    """
    Load documents from URLs or PDF directory and index into Pinecone.

    If no URLs are provided, defaults to indexing PDFs from settings.DATA_DIR.
    """
    settings = get_settings()
    if urls:
        docs = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
                logger.info(f"Loaded document from {url}")
            except Exception as e:
                logger.warning(f"Failed to load {url}: {e}")

        if docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            splits = text_splitter.split_documents(docs)
            global _vectorstore, _retriever
            _vectorstore = PineconeVectorStore.from_documents(
                documents=splits,
                embedding=_get_embeddings(),
                index_name=settings.PINECONE_INDEX_NAME,
            )
            _retriever = _vectorstore.as_retriever(
                search_kwargs={"k": settings.RETRIEVAL_K}
            )
            return len(splits)

    # Fall back to indexing local PDFs
    return load_and_index_pdfs()


def _load_existing_store() -> bool:
    """Try to connect to existing Pinecone index. Returns True if successful."""
    global _vectorstore, _retriever

    settings = get_settings()
    try:
        _vectorstore = PineconeVectorStore.from_existing_index(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=_get_embeddings(),
        )
        _retriever = _vectorstore.as_retriever(
            search_kwargs={"k": settings.RETRIEVAL_K}
        )
        logger.info(
            f"Successfully connected to existing Pinecone index: "
            f"'{settings.PINECONE_INDEX_NAME}'"
        )
        return True
    except Exception as e:
        logger.warning(f"Could not connect to Pinecone index: {e}")
        return False


def get_retriever() -> VectorStoreRetriever:
    """
    Return the Pinecone vectorstore retriever.

    On first call, connects to existing Pinecone index.
    If connection fails or index is not initialized, indexes PDFs from settings.DATA_DIR.
    """
    global _retriever

    if _retriever is not None:
        return _retriever

    # Try loading existing store first
    if _load_existing_store():
        assert _retriever is not None
        return _retriever

    # No existing store — index PDF documents
    logger.info("Connecting/indexing PDF documents into Pinecone...")
    load_and_index_pdfs()

    if _retriever is None:
        raise RuntimeError("Failed to initialize Pinecone vectorstore retriever")

    return _retriever


def is_ready() -> bool:
    """Check if the vectorstore is initialized and ready."""
    return _retriever is not None

