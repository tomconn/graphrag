"""GraphRAG ingest pipeline (one-off job).

Modules:
    parse    — walk data/, classify by parent dir, read files
    chunk    — class-specific chunking (code / markdown / regulatory)
    embed    — FastEmbed dense embeddings
    extract  — schema-guided entity/relation extraction (LLM)
    write    — Neo4j writes per docs/contracts.md
    main     — orchestration
"""