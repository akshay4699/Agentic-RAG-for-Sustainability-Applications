"""
Helper script to create/re-create Pinecone index with correct dimensions (2048).

Usage:
    cd agentic_rag_app
    python -m backend.create_pinecone_index
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinecone import Pinecone, ServerlessSpec
from backend.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    settings = get_settings()
    api_key = settings.PINECONE_API_KEY
    index_name = settings.PINECONE_INDEX_NAME

    if not api_key:
        logger.error("PINECONE_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name in existing_indexes:
        logger.info(f"Deleting existing Pinecone index '{index_name}'...")
        pc.delete_index(index_name)
        time.sleep(5)  # wait for deletion

    logger.info(f"Creating Pinecone index '{index_name}' with dimension=2048 and metric='cosine'...")
    pc.create_index(
        name=index_name,
        dimension=2048,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    logger.info(f"Successfully created Pinecone index '{index_name}' (dimension 2048)!")


if __name__ == "__main__":
    main()
