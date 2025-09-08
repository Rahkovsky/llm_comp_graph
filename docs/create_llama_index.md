## create_llama_index.py — Technical Overview

### Purpose
Build a local semantic search index over SEC 10-K texts using LlamaIndex and Chroma, with optional metadata-only mode.

### Inputs
- `--input-dir` (default: OUTDIR_10K): Root folder of 10-K `.txt` files.
- `--file-pattern` (default: `*.txt`): Files to include.
- `--output-dir` (default: `data/llama_index`): Persisted index location.
- `--chunk-size` (default: 900): Desired text chunk length.
- `--chunk-overlap` (default: 160): Desired overlap between chunks.
- `--max-files` (optional): Cap number of files.
- `--resolve-company-names` (default: true): Fetch company_name via SEC by CIK.
- `--embedding-model` (default: `BAAI/bge-base-en-v1.5`): HuggingFace embedding model.
- `--collection-name` (default: `10k_documents`): Chroma collection name.
- `--rebuild` (flag): Clear existing Chroma/index dirs before building.
- `--metadata-only` (flag): Build only `metadata_index.json` without vector index.

### Outputs
- Vector index persisted under `<output-dir>/chroma_db` and `<output-dir>/index` (unless `--metadata-only`).
- `metadata_index.json` summarizing per-ticker stats and file paths.
- `embedding_meta.json` recording effective chunking and embedding settings.

### Processing Pipeline
1) Discover files: `input_dir` + `file_pattern` (recursive).
2) Metadata extraction: parse filename into `ticker`, `cik`, `filing_date`, `accession`, store `file_path` and `relative_path`.
3) Optional name resolution: CIK → company_name via SEC submissions API (cached).
4a) Metadata-only mode: read text, produce `Document`-like items (or dicts), then aggregate into `metadata_index.json`.
4b) Streaming mode: for each file, read text → create `Document` → split into nodes → insert into Chroma via LlamaIndex; collect lightweight metadata in-memory for `metadata_index.json`.
5) Persist index (streaming mode) and write sidecar metadata files.

### Chunking Heuristics
The script bounds user-provided chunking by the embedding model capacity to avoid truncation at embed time.

- Effective chunk size:
  `eff_chunk = max(DEFAULT_MIN_CHUNK, min(chunk_size, model_max - DEFAULT_MODEL_MARGIN))`
- Effective overlap:
  `eff_overlap = max(DEFAULT_MIN_OVERLAP, min(int(DEFAULT_OVERLAP_RATIO * eff_chunk), eff_chunk // 2))`

Current defaults (centralized in the script):
- `DEFAULT_MIN_CHUNK = 64`
- `DEFAULT_MODEL_MARGIN = 32`  (reserve tokens for safety)
- `DEFAULT_MIN_OVERLAP = 32`
- `DEFAULT_OVERLAP_RATIO = 0.18`

Notes:
- For BGE/E5 models, the script sets appropriate query/text instructions when applicable.
- Overlap is bounded to half of the effective chunk to avoid degenerate overlap.
- In streaming mode, memory stays bounded since full corpora are not held in RAM.

### Error Handling
- File IO: guarded reads; skips empty/unreadable files.
- SEC name resolution: network exceptions are contained; falls back gracefully.
- Rebuild: removes existing Chroma/index directories only when `--rebuild` is provided.

### Example
```bash
python scripts/create_llama_index.py \
  --input-dir data/input/10K \
  --output-dir llama_index \
  --chunk-size 900 --chunk-overlap 160 \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --rebuild
```
