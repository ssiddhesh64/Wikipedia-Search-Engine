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