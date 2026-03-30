from collections import Counter, defaultdict
import math
import re
import sqlite3
from nltk.corpus import stopwords


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
        self.stop_words = set(stopwords.words("english"))

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
        while i < 2:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            for article_id, title, section_text in rows:
                text = title + " " + (section_text or "")
                tokens = self._process(text)

                self.doc_store[article_id] = title
                self.doc_len[article_id] += sum(
                    len(tokens)
                )  # Store document length for TF-IDF normalization

                for pos, token in enumerate(tokens):
                    self.pos_index[token][article_id].append(pos)

                # for token, count in freq.items():
                #     self.pos_index[token][article_id]
                freq = Counter(tokens)
                if article_id not in seen_docs:
                    seen_docs.add(article_id)
                    self.doc_count += 1

                    for token in freq.keys():
                        self.doc_freq[token] += 1

            i += 1
            # print(f"Processed {i * 1000} articles...")
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
            for article_id, tf in self.pos_index.get(token, {}).items():
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
    search_engine.build_index()

    results = search_engine.search_simple("machine learning")
    print("\nSearch Results:")
    search_engine.print_results(results)
