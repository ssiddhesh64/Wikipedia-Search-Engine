"""
seed_data.py

Seeds the SQLite database with real Wikipedia articles so the search engine
can be run and tested immediately without requiring the full 21GB dump.
"""

import argparse
import json
import sqlite3
import sys
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
    "TF-IDF",
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

# Fallback articles in case of offline execution or network restriction
FALLBACK_ARTICLES = [
    (
        1,
        "Machine learning",
        "Introduction",
        "Machine learning is a field of inquiry devoted to understanding and building methods that learn from data. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision.",
    ),
    (
        2,
        "Artificial intelligence",
        "Definition",
        "Artificial intelligence is the intelligence of machines or software, as opposed to the intelligence of living beings, primarily of humans. It is a field of study in computer science that develops and studies intelligent machines, commonly referred to as AI.",
    ),
    (
        3,
        "Information retrieval",
        "Overview",
        "Information retrieval is the science of searching for information in a document, searching for documents themselves, and also searching for the metadata that describes data, and for databases of texts, images or sounds. Automated information retrieval systems are used to reduce information overload.",
    ),
    (
        4,
        "Inverted index",
        "Data Structure",
        "In computer science, an inverted index is a database index storing a mapping from content, such as words or numbers, to its locations in a document or a set of documents. The purpose of an inverted index is to allow fast full-text searches, at the cost of increased processing when a document is added to the database.",
    ),
    (
        5,
        "TF-IDF",
        "Definition",
        "In information retrieval, tf-idf, TF*IDF, or TFIDF, short for term frequency-inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling.",
    ),
    (
        6,
        "Okapi BM25",
        "Ranking Function",
        "In information retrieval, Okapi BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query. It is based on the probabilistic retrieval framework developed in the 1970s and 1980s by Stephen E. Robertson, Karen Spärck Jones, and others.",
    ),
    (
        7,
        "Quantum computing",
        "Principles",
        "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers. Quantum computers use qubits, which can exist in multiple states simultaneously due to superposition and entanglement.",
    ),
    (
        8,
        "Distributed computing",
        "Concepts",
        "Distributed computing is a field of computer science that studies distributed systems, whose components are located on different networked computers, which communicate and coordinate their actions by passing messages to one another from any system.",
    ),
    (
        9,
        "Python (programming language)",
        "Design",
        "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.",
    ),
    (
        10,
        "Search engine",
        "Mechanism",
        "A search engine is a software system designed to carry out web searches. They search the World Wide Web in a systematic way for particular information specified in a textual web search query. The search results are generally presented in a line of results, often referred to as search engine results pages.",
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


def seed_database(db_path="enwiki-20170820.db", target_count=40, use_network=True):
    print(f"Connecting to database: {db_path}...")
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    cursor = conn.cursor()

    # Check existing count
    cursor.execute("SELECT COUNT(*) FROM ARTICLES")
    existing_count = cursor.fetchone()[0]
    if existing_count > 0:
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
                print(f"Could not fetch '{topic}' online ({e}), falling back...")
                break

    # If network failed or was disabled, insert fallback curated articles
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
    args = parser.parse_args()

    seed_database(db_path=args.db, target_count=args.count, use_network=not args.offline)
