#!/usr/bin/env python3
"""Create a proper LlamaIndex for downloaded 10-K documents with content ingestion."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from llama_index import (
        VectorStoreIndex,
        Document,
        ServiceContext,
        StorageContext,
    )
    from llama_index.embeddings import HuggingFaceEmbedding
    from llama_index.node_parser import SentenceSplitter
    from llama_index.vector_stores import ChromaVectorStore
    import chromadb

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("⚠️  LlamaIndex not available. Install with: pip install llama-index chromadb")


def check_llama_index():
    """Check if LlamaIndex is available."""
    if not LLAMAINDEX_AVAILABLE:
        print("❌ LlamaIndex is required but not installed.")
        print("💡 Install with: pip install llama-index chromadb")
        return False
    return True


def get_file_metadata(file_path: Path) -> Dict:
    """Extract metadata from 10-K file path."""
    filename = file_path.stem
    parts = filename.split("_")

    if len(parts) >= 4:
        ticker = parts[0]
        cik = parts[1]
        date_str = parts[2]
        accession = "_".join(parts[3:])
    else:
        ticker = parts[0] if parts else "UNKNOWN"
        cik = parts[1] if len(parts) > 1 else "UNKNOWN"
        date_str = parts[2] if len(parts) > 2 else "UNKNOWN"
        accession = "_".join(parts[3:]) if len(parts) > 3 else "UNKNOWN"

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "filing_date": date_str,
        "accession": accession,
        "file_path": str(file_path),
        "relative_path": str(file_path.relative_to(Path.cwd())),
    }


def create_documents_with_metadata(
    base_dir: str, file_pattern: str = "*.txt"
) -> List[Document]:
    """Create LlamaIndex Document objects with metadata."""
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
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                print(f"  ⚠️  Empty file: {file_path}")
                continue

            # Extract metadata
            metadata = get_file_metadata(file_path)

            # Create LlamaIndex Document
            doc = Document(text=content, metadata=metadata)
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

    # Initialize embedding model (using a lightweight one)
    print("🤖 Loading embedding model...")
    try:
        embedding_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="models/embeddings",
        )
    except Exception as e:
        print(f"⚠️  Could not load embedding model: {e}")
        print("💡 Using default embeddings...")
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


def create_metadata_index(documents: List[Document]) -> Dict:
    """Create a metadata index for quick lookups."""
    print("📋 Creating metadata index...")

    # Group by company
    companies = {}
    for doc in documents:
        ticker = doc.metadata["ticker"]
        if ticker not in companies:
            companies[ticker] = {
                "ticker": ticker,
                "cik": doc.metadata["cik"],
                "total_filings": 0,
                "filing_dates": [],
                "accessions": [],
                "file_paths": [],
            }

        company = companies[ticker]
        company["total_filings"] += 1
        company["filing_dates"].append(doc.metadata["filing_date"])
        company["accessions"].append(doc.metadata["accession"])
        company["file_paths"].append(doc.metadata["file_path"])

    # Sort companies by filing count
    sorted_companies = sorted(
        companies.values(), key=lambda x: x["total_filings"], reverse=True
    )

    return {
        "companies": sorted_companies,
        "total_companies": len(sorted_companies),
        "total_documents": len(documents),
        "created_at": datetime.now().isoformat(),
    }


def save_metadata_index(metadata_index: Dict, output_file: str):
    """Save metadata index to JSON."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metadata_index, f, indent=2, ensure_ascii=False)
        print(f"💾 Metadata index saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving metadata index: {e}")


def print_summary(metadata_index: Dict):
    """Print summary of the created index."""
    print("\n" + "=" * 80)
    print("📊 LLAMA INDEX CREATION SUMMARY")
    print("=" * 80)
    print(f"Total Companies: {metadata_index['total_companies']}")
    print(f"Total Documents: {metadata_index['total_documents']}")
    print(f"Created At: {metadata_index['created_at']}")

    print("\n🏆 Top 10 Companies by Filing Count:")
    for i, company in enumerate(metadata_index["companies"][:10], 1):
        print(
            f"  {i:2d}. {company['ticker']:6s} - {company['total_filings']:3d} filings"
        )

    print("\n💡 Your documents are now searchable using:")
    print("   - Vector similarity search")
    print("   - Metadata filtering")
    print("   - RAG (Retrieval-Augmented Generation)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Create a proper LlamaIndex for 10-K documents with content ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create index for all downloaded documents
  python scripts/create_llama_index_advanced.py

  # Create index for specific directory
  python scripts/create_llama_index_advanced.py --input-dir data/input/10K/ALL_COMPANIES/2024

  # Custom output directory
  python scripts/create_llama_index_advanced.py --output-dir llama_index_2024

  # Custom chunk size
  python scripts/create_llama_index_advanced.py --chunk-size 2048 --chunk-overlap 100

  # Include only specific file types
  python scripts/create_llama_index_advanced.py --file-pattern "*.plain.txt"
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
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Create only metadata index (skip vector index)",
    )

    args = parser.parse_args()

    # Check prerequisites
    if not args.metadata_only and not check_llama_index():
        return

    print("🚀 Creating Advanced LLAMA Index for 10-K Documents")
    print(f"📂 Input Directory: {args.input_dir}")
    print(f"📄 File Pattern: {args.file_pattern}")
    print(f"📁 Output Directory: {args.output_dir}")

    if not args.metadata_only:
        print(f"🔧 Chunk Size: {args.chunk_size}")
        print(f"🔄 Chunk Overlap: {args.chunk_overlap}")

    # Create documents with metadata
    start_time = time.time()
    documents = create_documents_with_metadata(args.input_dir, args.file_pattern)

    if not documents:
        print("❌ No documents found. Check your input directory and file pattern.")
        return

    # Create metadata index
    metadata_index = create_metadata_index(documents)
    metadata_file = os.path.join(args.output_dir, "metadata_index.json")
    save_metadata_index(metadata_index, metadata_file)

    # Create vector index (if requested)
    if not args.metadata_only:
        create_vector_index(
            documents, args.output_dir, args.chunk_size, args.chunk_overlap
        )

    # Print summary
    print_summary(metadata_index)

    # Performance stats
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Index creation completed in {elapsed_time:.2f} seconds")

    if not args.metadata_only:
        print("🔍 Vector index ready for semantic search and RAG")
        print("📊 Metadata index available for quick lookups")
    else:
        print("📊 Metadata-only index created")


if __name__ == "__main__":
    main()
