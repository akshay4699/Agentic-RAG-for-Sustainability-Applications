"""
VectorStore service — handles document loading, splitting, embedding, and retrieval.

Uses Chroma with persistence so the index survives restarts.
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singletons
_vectorstore: Chroma | None = None
_retriever: VectorStoreRetriever | None = None


def _get_embeddings() -> NVIDIAEmbeddings:
    """Return the configured embeddings model."""
    settings = get_settings()
    return NVIDIAEmbeddings(model=settings.EMBEDDING_MODEL)


def load_and_index_documents(urls: list[str] | None = None) -> int:
    """
    Load documents from URLs, split them, and index into Chroma.

    Args:
        urls: List of URLs to load. Falls back to settings defaults.

    Returns:
        Number of chunks indexed.
    """
    global _vectorstore, _retriever

    settings = get_settings()
    urls = urls or settings.DOCUMENT_URLS

    # --- Load documents ---
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
            logger.info(f"Loaded document from {url}")
        except Exception as e:
            logger.warning(f"Failed to load {url}: {e}")

    if not docs:
        logger.error("No documents loaded!")
        return 0

    logger.info(f"Loaded {len(docs)} documents total")

    # --- Split into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(splits)} chunks")

    # --- Create vectorstore ---
    persist_dir = Path(settings.CHROMA_PERSIST_DIR).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    _vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=_get_embeddings(),
        collection_name=settings.VECTORSTORE_COLLECTION,
        persist_directory=str(persist_dir),
    )
    _retriever = _vectorstore.as_retriever(
        search_kwargs={"k": settings.RETRIEVAL_K}
    )

    logger.info(f"Vector store created with {len(splits)} chunks at {persist_dir}")
    return len(splits)


def _load_existing_store() -> bool:
    """Try to load an existing persisted Chroma store. Returns True if successful."""
    global _vectorstore, _retriever

    settings = get_settings()
    persist_dir = Path(settings.CHROMA_PERSIST_DIR).resolve()

    if not persist_dir.exists():
        return False

    try:
        _vectorstore = Chroma(
            collection_name=settings.VECTORSTORE_COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=str(persist_dir),
        )
        # Check if the collection actually has documents
        if _vectorstore._collection.count() == 0:
            logger.info("Persisted store exists but is empty — needs indexing")
            return False

        _retriever = _vectorstore.as_retriever(
            search_kwargs={"k": settings.RETRIEVAL_K}
        )
        logger.info(
            f"Loaded existing vectorstore with "
            f"{_vectorstore._collection.count()} documents"
        )
        return True
    except Exception as e:
        logger.warning(f"Could not load persisted store: {e}")
        return False


def get_retriever() -> VectorStoreRetriever:
    """
    Return the vectorstore retriever.

    On first call, tries to load a persisted store.
    If none exists, indexes the default documents.
    """
    global _retriever

    if _retriever is not None:
        return _retriever

    # Try loading existing store first
    if _load_existing_store():
        assert _retriever is not None
        return _retriever

    # No existing store — index default documents
    logger.info("No existing vectorstore found — indexing default documents...")
    load_and_index_documents()

    if _retriever is None:
        raise RuntimeError("Failed to initialize vectorstore retriever")

    return _retriever


def is_ready() -> bool:
    """Check if the vectorstore is initialized and ready."""
    return _retriever is not None
