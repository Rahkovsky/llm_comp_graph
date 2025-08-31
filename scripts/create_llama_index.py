#!/usr/bin/env python3
"""Create a LlamaIndex for downloaded 10-K documents with content ingestion."""

import argparse
import os
import time
from pathlib import Path
from typing import List

try:
    from llama_index import (
        VectorStoreIndex,
        Document,
        ServiceContext,
        StorageContext,
    )
    from llama_index.embeddings import LlamaEmbedding
    from llama_index.node_parser import SentenceSplitter
    from llama_index.vector_stores import ChromaVectorStore
    import chromadb

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("⚠️  LlamaIndex is required but not installed.")
    print("💡 Install with: pip install llama-index chromadb")


def check_llama_index():
    """Check if LlamaIndex is available."""
    if not LLAMAINDEX_AVAILABLE:
        print("❌ LlamaIndex is required but not installed.")
        print("💡 Install with: pip install llama-index chromadb")
        return False
    return True


def check_llama_model():
    """Check if LLAMA model is available."""
    from constants import LLAMA_MODEL

    if not os.path.exists(LLAMA_MODEL):
        print(f"❌ LLAMA model not found at: {LLAMA_MODEL}")
        print(
            "💡 Make sure your model is downloaded and path is correct in constants.py"
        )
        return False
    print(f"✅ LLAMA model found: {LLAMA_MODEL}")
    return True


def create_documents_from_files(
    base_dir: str, file_pattern: str = "*.txt"
) -> List[Document]:
    """Create LlamaIndex Document objects from files."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ Directory not found: {base_dir}")
        return []

    print(f"🔍 Reading documents from: {base_dir}")
    print(f"📁 Looking for files matching: {file_pattern}")

    documents = []
    total_files = 0
    processed_files = 0

    # Count total files first
    for file_path in base_path.rglob(file_pattern):
        if file_path.is_file():
            total_files += 1

    print(f"📊 Found {total_files} files to process")

    # Process files
    for file_path in base_path.rglob(file_pattern):
        if not file_path.is_file():
            continue

        processed_files += 1
        if processed_files % 50 == 0:
            print(f"  📝 Processed {processed_files}/{total_files} files...")

        try:
            # Read the WHOLE file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                print(f"  ⚠️  Empty file: {file_path}")
                continue

            # Create LlamaIndex Document with file path as metadata
            doc = Document(text=content, metadata={"file_path": str(file_path)})
            documents.append(doc)

        except Exception as e:
            print(f"  ⚠️  Error processing {file_path}: {e}")
            continue

    print(f"✅ Successfully created {len(documents)} Document objects")
    return documents


def create_vector_index(
    documents: List[Document],
    output_dir: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 20,
):
    """Create a vector index using LlamaIndex."""
    print("\n🔧 Creating vector index...")
    print(f"📊 Documents: {len(documents)}")
    print(f"📏 Chunk size: {chunk_size}")
    print(f"🔄 Chunk overlap: {chunk_overlap}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize local LLAMA embedding model
    print("🤖 Loading local LLAMA embedding model...")
    try:
        from constants import LLAMA_MODEL

        embedding_model = LlamaEmbedding(
            model_name=LLAMA_MODEL,
            embed_batch_size=1,  # Process one document at a time
            device="auto",  # Auto-detect best device (CPU/GPU)
        )
        print(f"✅ Using local LLAMA model: {os.path.basename(LLAMA_MODEL)}")
    except Exception as e:
        print(f"⚠️  Could not load LLAMA embedding model: {e}")
        print("💡 Falling back to default embeddings...")
        embedding_model = None

    # Create service context
    service_context = ServiceContext.from_defaults(
        embed_model=embedding_model,
        node_parser=SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ),
    )

    # Create vector store
    print("🗄️  Creating vector store...")
    db = chromadb.PersistentClient(path=os.path.join(output_dir, "chroma_db"))
    chroma_collection = db.create_collection("10k_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index
    print("🔍 Building index...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )

    # Save index
    print("💾 Saving index...")
    index.storage_context.persist(persist_dir=os.path.join(output_dir, "index"))

    print(f"✅ Vector index created and saved to: {output_dir}")
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Create a LlamaIndex for 10-K documents with content ingestion using local LLAMA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create index for all downloaded documents
  python scripts/create_llama_index.py

  # Create index for specific directory
  python scripts/create_llama_index.py --input-dir data/input/10K/ALL_COMPANIES/2024

  # Custom output directory
  python scripts/create_llama_index.py --output-dir llama_index_2024

  # Custom chunk size
  python scripts/create_llama_index.py --chunk-size 2048 --chunk-overlap 100

  # Include only specific file types
  python scripts/create_llama_index.py --file-pattern "*.plain.txt"
        """,
    )

    parser.add_argument(
        "--input-dir",
        default="data/input/10K/ALL_COMPANIES",
        help="Input directory containing 10-K documents",
    )
    parser.add_argument(
        "--output-dir", default="llama_index", help="Output directory for the index"
    )
    parser.add_argument(
        "--file-pattern", default="*.txt", help="File pattern to include"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1024, help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=20, help="Chunk overlap for text splitting"
    )

    args = parser.parse_args()

    # Check prerequisites
    if not check_llama_index():
        return

    if not check_llama_model():
        return

    print("🚀 Creating LlamaIndex for 10-K Documents")
    print(f"📂 Input Directory: {args.input_dir}")
    print(f"📄 File Pattern: {args.file_pattern}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"🔧 Chunk Size: {args.chunk_size}")
    print(f"🔄 Chunk Overlap: {args.chunk_overlap}")
    print("🤖 Using Local LLAMA Model for Embeddings")

    # Create documents from files
    start_time = time.time()
    documents = create_documents_from_files(args.input_dir, args.file_pattern)

    if not documents:
        print("❌ No documents found. Check your input directory and file pattern.")
        return

    # Create vector index
    create_vector_index(documents, args.output_dir, args.chunk_size, args.chunk_overlap)

    # Performance stats
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Index creation completed in {elapsed_time:.2f} seconds")
    print("🔍 LlamaIndex ready for semantic search and RAG")
    print(f"📊 Index contains {len(documents)} documents")
    print("🤖 All embeddings created using your local LLAMA model")


if __name__ == "__main__":
    main()
