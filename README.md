# Wikipedia Search Engine

A simple search engine built on top of a Wikipedia dataset, demonstrating core information retrieval concepts like inverted indexing and text preprocessing.

---

## Features

- Search Wikipedia articles by keyword  
- Inverted index for fast lookup  
- Text preprocessing:
  - Lowercasing  
  - Tokenization  
  - Stopword removal  
- Basic ranking using term frequency  

---

## Architecture

```mermaid
flowchart LR

    %% -------------------
    %% Indexing Pipeline
    %% -------------------
    subgraph Indexing Flow
        A["SQLite Dump (21GB)"] --> B["Batch Processing (Streaming)"]
        B --> C["Text Preprocessing"]
        C --> D["Inverted Index (In-Memory)"]
        D --> E["Persist to Disk"]
    end

    %% -------------------
    %% Query Pipeline
    %% -------------------
    subgraph Query Flow
        Q1["User Query"] --> Q2["Query Preprocessing"]
        Q2 --> Q3["Load Index / Lookup"]
        Q3 --> Q4["Ranking Algorithm"]
        Q4 --> Q5["Top-K Results"]
    end

    %% Optional connection
    E --> Q3
```

---

## Tech Stack

- Python  
- SQLite  

---

## Dataset

- Wikipedia dump (`enwiki-20170820.db`)

### Schema:
```sql
CREATE TABLE ARTICLES (
  ARTICLE_ID INTEGER,
  TITLE TEXT,
  SECTION_TITLE TEXT,
  SECTION_TEXT TEXT
);
```

## Next Steps

### Search Quality Improvements
- Implement **TF-IDF / BM25 ranking** for better relevance  

### Performance & Scalability
- Move from in-memory index to a **disk-based inverted index** (optimized for 21GB+)  
- Implement **caching layer** (e.g., Redis) for frequent queries  
- Enable **parallel indexing** for faster preprocessing  

### Advanced Features
- Add **autocomplete / typeahead suggestions**  
- Implement **highlighting of matched terms** in results  
- Support **filters** (by section, title, etc.)  


## 📦 Index Storage Format

The search engine uses a **disk-based inverted index** to enable efficient querying without loading the entire dataset into memory.

---

### 🗂️ 1. `metadata.pkl`

Stores global metadata required for ranking and retrieval.

```python
{
  "doc_len": {doc_id: int},        # Total terms in each document
  "doc_freq": {token: int},        # Number of documents containing the token
  "doc_store": {doc_id: title},    # Mapping of document IDs to titles
  "doc_count": int                 # Total number of documents
}
```

#### postings.bin structure:
```
[token postings]
number_of_docs (int)
  For each document:
    → doc_id (int)
    → term_frequency (int)
    → positions_count (int)
    → positions (list of ints)
```

#### vokab.pkl
Stores token -> offsets in postings files
```
{
  "machine": 1024,
  "learning": 2048,
  ...
}
```