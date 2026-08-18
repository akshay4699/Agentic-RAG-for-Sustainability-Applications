"""
Pinecone Ingestion Script

Run this script to load, split, and index all PDF documents in the `data/` folder
into your Pinecone index.

Usage:
    cd agentic_rag_app
    python -m backend.ingest_pinecone
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import get_settings
from backend.services.vectorstore import load_and_index_pdfs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion runner."""
    logger.info("Starting Pinecone Document Ingestion...")
    settings = get_settings()

    logger.info(f"Target Pinecone Index: {settings.PINECONE_INDEX_NAME}")
    logger.info(f"PDF Data Directory:    {settings.DATA_DIR}")

    try:
        indexed_count = load_and_index_pdfs()
        if indexed_count > 0:
            logger.info(
                f"Successfully indexed {indexed_count} text chunks into Pinecone index '{settings.PINECONE_INDEX_NAME}'!"
            )
        else:
            logger.warning("No documents were indexed. Please check your data/ directory.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
