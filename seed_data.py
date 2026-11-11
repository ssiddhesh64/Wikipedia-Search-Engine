"""
seed_data.py

Seeds the SQLite database with real Wikipedia articles so the search engine
can be run and tested immediately without requiring the full 21GB dump.
"""

import argparse
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request

DEFAULT_TOPICS = [
    "Machine learning",
    "Artificial intelligence",
    "Deep learning",
    "Natural language processing",
    "Computer vision",
    "Information retrieval",
    "Search engine",
    "PageRank",
    "Inverted index",
    "Tf–idf",
    "Okapi BM25",
    "Quantum computing",
    "Operating system",
    "Distributed computing",
    "Database",
    "Relational database",
    "NoSQL",
    "World Wide Web",
    "Cloud computing",
    "Computer network",
    "Internet protocol suite",
    "Algorithm",
    "Data structure",
    "Binary search tree",
    "Graph theory",
    "Compiler",
    "Python (programming language)",
    "C (programming language)",
    "Linux",
    "Alan Turing",
    "Ada Lovelace",
    "Claude Shannon",
    "Solar System",
    "Quantum mechanics",
    "General relativity",
    "DNA",
    "CRISPR gene editing",
    "Industrial Revolution",
    "Renaissance",
    "Apollo program",
]

# Rich fallback articles with cross-topic overlap
FALLBACK_ARTICLES = [
    (
        1,
        "Machine learning",
        "Introduction",
        "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Machine learning algorithms build a mathematical model based on sample training data to make predictions or decisions. Deep learning, supervised learning, unsupervised learning, and reinforcement learning are core paradigms. Applications include computer vision, natural language processing, search engines, and automated reasoning.",
    ),
    (
        2,
        "Artificial intelligence",
        "Overview",
        "Artificial intelligence (AI) is the intelligence of machines and computers. Core subfields of artificial intelligence include machine learning, deep learning, computer vision, natural language processing, and robotics. Modern artificial intelligence heavily relies on deep neural networks trained on large corpora of text and images. Prominent applications of AI include advanced web search engines, autonomous vehicles, speech recognition, and medical diagnostics.",
    ),
    (
        3,
        "Deep learning",
        "Architecture",
        "Deep learning is a subset of machine learning based on artificial neural networks with representation learning. Deep learning architectures such as deep neural networks, convolutional neural networks (CNNs), and transformers have produced state-of-the-art results in computer vision, speech recognition, natural language processing, and audio recognition. Training deep models requires large datasets, gradient descent, and high-performance computing hardware.",
    ),
    (
        4,
        "Natural language processing",
        "Methods",
        "Natural language processing (NLP) is an interdisciplinary subfield of computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human language. Modern NLP algorithms are based on machine learning, particularly deep neural networks and transformer language models. Common tasks include text classification, sentiment analysis, machine translation, information retrieval, and search engine query understanding.",
    ),
    (
        5,
        "Computer vision",
        "Techniques",
        "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images to produce numerical or symbolic information. Deep learning and convolutional neural networks are dominant in image classification, object detection, and visual search engines.",
    ),
    (
        6,
        "Information retrieval",
        "Concepts",
        "Information retrieval (IR) is the science of searching for information in a document, searching for documents themselves, and searching for metadata describing databases. Web search engines are the most prominent application of information retrieval. Key IR techniques include tokenization, stemming, inverted index construction, TF-IDF weighting, and Okapi BM25 probabilistic ranking functions.",
    ),
    (
        7,
        "Search engine",
        "Architecture",
        "A search engine is an information retrieval software system designed to carry out searches across documents, databases, or the World Wide Web. Search engines use web crawlers to collect documents, text processors to tokenize and normalize text, inverted index structures for fast candidate retrieval, and ranking algorithms such as PageRank and Okapi BM25 to order results by relevance.",
    ),
    (
        8,
        "PageRank",
        "Graph Algorithm",
        "PageRank is an algorithm used by Google Search to rank web pages in search engine results. PageRank works by counting the number and quality of links to a page to estimate a rough measure of how important the website is. It models a random surfer browsing a directed graph representing the World Wide Web, calculating the stationary probability distribution across nodes.",
    ),
    (
        9,
        "Inverted index",
        "Data Structure",
        "In computer science, an inverted index is a database index data structure storing a mapping from words or tokens to their locations in a document or database. Inverted indices are the primary data structure used by search engines and information retrieval systems to achieve sub-millisecond query responses across billions of documents. Positional inverted indices also store word offsets for exact phrase search.",
    ),
    (
        10,
        "TF-IDF",
        "Weighting Scheme",
        "In information retrieval, TF-IDF (term frequency-inverse document frequency) is a numerical statistic reflecting how important a word is to a document within a collection or corpus. Search engines frequently use TF-IDF variants to score document relevance against multi-word user queries, balancing term frequency within a document against corpus-wide rarity.",
    ),
    (
        11,
        "Okapi BM25",
        "Ranking Function",
        "Okapi BM25 is a state-of-the-art probabilistic ranking function widely used by search engines to estimate the relevance of documents to search queries. BM25 improves upon classical TF-IDF by incorporating document length normalization and non-linear term frequency saturation, preventing excessively long documents from unfairly dominating search results.",
    ),
    (
        12,
        "Database",
        "Systems",
        "A database is an organized collection of data stored and accessed electronically from a computer system. Database management systems (DBMS) such as PostgreSQL, MySQL, and SQLite provide mechanisms for data querying, indexing, transaction processing, and concurrency control. Modern databases support relational schemas, full-text inverted indices, and distributed replication.",
    ),
    (
        13,
        "Distributed computing",
        "Architecture",
        "Distributed computing is a field of computer science that studies distributed systems consisting of autonomous computing nodes that communicate over a network. Large-scale search engines, cloud databases, and machine learning clusters rely on distributed computing frameworks to partition inverted index postings, scale storage, and parallelize ranking computations.",
    ),
    (
        14,
        "Quantum computing",
        "Principles",
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, physics, and mathematics that utilizes quantum mechanics to solve complex problems faster than classical computers. The field includes quantum hardware development and quantum algorithm design, using qubits that exhibit superposition and quantum entanglement.",
    ),
    (
        15,
        "Python (programming language)",
        "Ecosystem",
        "Python is a high-level, general-purpose programming language widely adopted for data science, machine learning, artificial intelligence, and web development. Python features a rich standard library and extensive package ecosystem, making it a popular choice for prototyping search engines, natural language processing pipelines, and data structures.",
    ),
    (
        16,
        "Data structure",
        "Algorithms",
        "In computer science, a data structure is a data organization, management, and storage format that enables efficient access and modification. Common data structures include arrays, linked lists, hash tables, trees, and graphs. Search engines rely heavily on hash tables, inverted indices, and tries for rapid keyword lookups.",
    ),
    (
        17,
        "World Wide Web",
        "History",
        "The World Wide Web is an information system where documents and web resources are identified by URLs and accessible over the Internet. The rapid growth of web pages created an urgent need for automated search engines, web crawlers, and information retrieval algorithms capable of indexing billions of hyperlinked HTML documents.",
    ),
    (
        18,
        "Operating system",
        "Kernel",
        "An operating system (OS) is system software that manages computer hardware, software resources, and provides common services for computer programs. Core components include the kernel, file system, memory management, and process scheduling. High-performance databases and search engines optimize I/O using OS page caches, file seeks, and memory-mapped files.",
    ),
    (
        19,
        "Algorithm",
        "Complexity",
        "In mathematics and computer science, an algorithm is a finite sequence of rigorous instructions used to perform a computation or solve a problem. Algorithms are evaluated by time complexity and space complexity using Big O notation. Search engines optimize query algorithms to achieve logarithmic or near-constant time retrieval.",
    ),
    (
        20,
        "Cloud computing",
        "Infrastructure",
        "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user. Major search engines run in large-scale cloud data centers distributed globally to provide low latency, redundancy, and high availability.",
    ),
]


def create_schema(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ARTICLES (
            ARTICLE_ID INTEGER PRIMARY KEY,
            TITLE TEXT NOT NULL,
            SECTION_TITLE TEXT,
            SECTION_TEXT TEXT
        )
        """
    )
    conn.commit()


def fetch_wikipedia_article(title):
    """Fetches plain text extract of a Wikipedia article using the Wikipedia API."""
    url = (
        "https://en.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": "1",
                "exintro": "0",
                "redirects": "1",
            }
        )
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "WikipediaSearchEngineDemo/1.0 (educational portfolio project)"},
    )

    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id != "-1" and "extract" in page:
                return page.get("title", title), page["extract"]
    return None, None


def seed_database(db_path="enwiki-20170820.db", target_count=40, use_network=True, force_clear=False):
    print(f"Connecting to database: {db_path}...")
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    cursor = conn.cursor()

    # Check existing count
    cursor.execute("SELECT COUNT(*) FROM ARTICLES")
    existing_count = cursor.fetchone()[0]
    if existing_count > 0:
        if force_clear:
            cursor.execute("DELETE FROM ARTICLES")
            conn.commit()
            print("Cleared existing articles.")
        else:
            print(f"Database already contains {existing_count} articles.")
            choice = input("Do you want to re-seed (clear existing articles)? [y/N]: ").strip().lower()
            if choice == "y":
                cursor.execute("DELETE FROM ARTICLES")
                conn.commit()
                print("Cleared existing articles.")
            else:
                print("Keeping existing data. Exiting.")
                conn.close()
                return

    inserted = 0
    article_id = 1

    if use_network:
        print(f"Fetching up to {target_count} articles from Wikipedia API...")
        for topic in DEFAULT_TOPICS[:target_count]:
            try:
                time.sleep(0.35)  # Respect Wikipedia rate limits
                title, extract = fetch_wikipedia_article(topic)
                if not extract:
                    continue

                # Store first main sections (capped to reasonable length)
                section_text = extract[:15000]
                cursor.execute(
                    """
                    INSERT INTO ARTICLES (ARTICLE_ID, TITLE, SECTION_TITLE, SECTION_TEXT)
                    VALUES (?, ?, ?, ?)
                    """,
                    (article_id, title, "Overview", section_text),
                )
                article_id += 1
                inserted += 1
                print(f"[{inserted}/{target_count}] Seeded: {title}")
                conn.commit()
            except Exception as e:
                print(f"Could not fetch '{topic}' ({e}), continuing...")
                continue

    # If network failed or offline mode was requested, insert fallback curated articles
    if inserted == 0:
        print("Using offline fallback curated article dataset...")
        for aid, title, sec_title, text in FALLBACK_ARTICLES:
            cursor.execute(
                """
                INSERT INTO ARTICLES (ARTICLE_ID, TITLE, SECTION_TITLE, SECTION_TEXT)
                VALUES (?, ?, ?, ?)
                """,
                (aid, title, sec_title, text),
            )
            inserted += 1
        conn.commit()
        print(f"Inserted {inserted} curated articles.")

    conn.close()
    print(f"\nDone! Database '{db_path}' successfully seeded with {inserted} articles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Wikipedia Search Engine SQLite database")
    parser.add_argument("--db", default="enwiki-20170820.db", help="Path to SQLite DB file")
    parser.add_argument("--count", type=int, default=40, help="Number of articles to fetch")
    parser.add_argument("--offline", action="store_true", help="Use offline curated dataset only")
    parser.add_argument("--force", "-f", action="store_true", help="Force clear existing database without prompt")
    args = parser.parse_args()

    seed_database(
        db_path=args.db,
        target_count=args.count,
        use_network=not args.offline,
        force_clear=args.force,
    )

