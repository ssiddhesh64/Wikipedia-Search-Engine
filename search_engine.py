from collections import Counter, defaultdict
import math
import re
import sqlite3
import pickle
import struct
import os

STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "at",
    "to",
    "of",
    "for",
    "with",
    "by",
    "as",
    "that",
    "this",
}


class SearchEngine:
    def __init__(self, db_path):
        self.db_path = db_path
        self.pos_index = defaultdict(lambda: defaultdict(list))
        self.doc_store = {}
        self.doc_freq = defaultdict(int)
        self.doc_count = 0
        self.doc_len = defaultdict(
            int
        )  # Store document lengths for TF-IDF normalization
        # self.stop_words = set(stopwords.words("english"))
        self.stop_words = set(STOP_WORDS)  # Override with custom stop words

    def index_exists(self, dir_path="index_data"):
        return (
            os.path.exists(os.path.join(dir_path, "vocab.pkl"))
            and os.path.exists(os.path.join(dir_path, "metadata.pkl"))
            and os.path.exists(os.path.join(dir_path, "postings.bin"))
        )

    def setup(self, query=None, dir_path="index_data", force=False):
        if self.index_exists(dir_path) and not force:
            print("Loading existing index...")
            self.load_index(query, dir_path)
        else:
            print("Building index from scratch...")
            self.build_index()

            if not hasattr(self, "pos_index"):
                raise Exception("pos_index not built. Required for disk index.")

            self._save_index_to_disk(dir_path)

    def _normalize(self, text):
        return text.lower()

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text)

    def _remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def _process(self, text):

        text = self._normalize(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stop_words(tokens)

        return tokens

    def _save_index_to_disk(self, dir_path="index_data"):

        os.makedirs(dir_path, exist_ok=True)

        vocab = {}
        offset = 0

        postings_path = os.path.join(dir_path, "postings.bin")
        vocab_path = os.path.join(dir_path, "vocab.pkl")
        metadata_path = os.path.join(dir_path, "metadata.pkl")

        with open(postings_path, "wb") as f:
            for token in sorted(self.pos_index.keys()):
                postings = self.pos_index[token]  # {doc_id: [positions]}

                vocab[token] = offset

                # number of documents
                f.write(struct.pack("I", len(postings)))
                offset += 4

                for doc_id, positions in postings.items():
                    tf = len(positions)

                    # doc_id + tf
                    f.write(struct.pack("II", doc_id, tf))
                    offset += 8

                    # positions count
                    f.write(struct.pack("I", tf))
                    offset += 4

                    # positions list
                    for pos in positions:
                        f.write(struct.pack("I", pos))
                        offset += 4

        # Save vocab
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

        # Save metadata
        metadata = {
            "doc_len": dict(self.doc_len),
            "doc_freq": dict(self.doc_freq),
            "doc_store": self.doc_store,
            "doc_count": self.doc_count,
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print("Index saved to disk!")
        f.close()

    def load_index(self, query=None, dir_path="index_data"):

        vocab_path = os.path.join(dir_path, "vocab.pkl")
        metadata_path = os.path.join(dir_path, "metadata.pkl")

        # Load vocab
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.doc_len = defaultdict(int, metadata["doc_len"])
        self.doc_freq = defaultdict(int, metadata["doc_freq"])
        self.doc_store = metadata["doc_store"]
        self.doc_count = metadata["doc_count"]

        print("Index loaded from disk!")
        f.close()

    def get_postings(self, token, dir_path="index_data"):
        if token not in self.vocab:
            return []

        postings_path = os.path.join(dir_path, "postings.bin")
        # Open postings file (keep open for fast reads)
        self.postings_file = open(postings_path, "rb")

        offset = self.vocab[token]
        self.postings_file.seek(offset)

        # Read number of documents
        num_docs = struct.unpack("I", self.postings_file.read(4))[0]

        postings = defaultdict(list)
        for _ in range(num_docs):
            doc_id, tf = struct.unpack("II", self.postings_file.read(8))

            pos_count = struct.unpack("I", self.postings_file.read(4))[0]

            positions = []
            for _ in range(pos_count):
                pos = struct.unpack("I", self.postings_file.read(4))[0]
                positions.append(pos)

            postings[doc_id] = positions

        self.postings_file.close()
        return postings

    def build_index(self):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT ARTICLE_ID, TITLE, SECTION_TEXT 
            FROM ARTICLES
        """
        )

        seen_docs = set()
        i = 0
        while i < 100:  # Limit to first 100k articles for testing
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            for article_id, title, section_text in rows:
                text = title + " " + (section_text or "")
                tokens = self._process(text)

                self.doc_store[article_id] = title
                self.doc_len[article_id] += len(
                    tokens
                )  # Store document length for TF-IDF normalization

                for pos, token in enumerate(tokens):
                    self.pos_index[token][article_id].append(pos)

                freq = Counter(tokens)
                if article_id not in seen_docs:
                    seen_docs.add(article_id)
                    self.doc_count += 1

                    for token in freq.keys():
                        self.doc_freq[token] += 1

            i += 1
            print(f"Processed {i * 1000} articles...")
        conn.close()

    def _idf(self, token):
        df = self.doc_freq.get(token, 0)
        if df == 0:
            return 0
        return (
            math.log((self.doc_count + 1) / (df + 1)) + 1
        )  # Smoothing to avoid division by zero

    def search_simple(self, query):
        query_tokens = self._process(query)
        results = defaultdict(int)

        for token in query_tokens:
            for article_id, count in self.pos_index.get(token, {}).items():
                results[article_id] += count

        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)

        return ranked[:10]

    def phrase_search(self, query):
        query_tokens = self._process(query)
        if not query_tokens:
            return []

        postings = [self.pos_index.get(token, {}) for token in query_tokens]

        common_docs = set(postings[0].keys())
        for p in postings[1:]:
            common_docs &= set(p.keys())

        results = []
        for doc_id in common_docs:
            positions_list = [p[doc_id] for p in postings]

            pos_sets = [set(positions) for positions in positions_list]
            for pos in positions_list[0]:
                match = True
                for i in range(1, len(pos_sets)):
                    next_positions = pos_sets[i]
                    if (pos + i) not in next_positions:
                        match = False
                        break
                if match:
                    results.append(doc_id)
                    break  # Stop after the first match in this document

        return results

    def search_tfidf(self, query):
        query_tokens = self._process(query)
        results = defaultdict(float)

        for token in query_tokens:
            idf = self._idf(token)
            postings = self.get_postings(token)
            for article_id, positions in postings.items():
                tf = len(positions)
                results[article_id] += (1 + math.log(tf)) * idf

        for article_id in results:
            results[article_id] /= self.doc_len[
                article_id
            ]  # Normalize by document length

        # phrase boost
        phrase_docs = self.phrase_search(query)
        for doc_id in phrase_docs:
            if doc_id in results:
                results[doc_id] *= 1.5  # Boost score for phrase matches

        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)

        return ranked[:10]

    def print_results(self, results):
        for article_id, score in results:
            print(
                f"Article ID: {article_id}, Title: {self.doc_store.get(article_id, 'Unknown')}, Score: {score}"
            )


if __name__ == "__main__":
    search_engine = SearchEngine("enwiki-20170820.db")

    query = "machine learning"
    search_engine.setup(query=query)  # Force rebuild index for testing
    results = search_engine.search_tfidf(query)
    print("\nSearch Results:")
    search_engine.print_results(results)
