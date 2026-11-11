"""
app.py

Zero-dependency interactive Web UI and REST API for the Wikipedia Search Engine.
Runs out of the box with Python standard library (http.server).
"""

import http.server
import json
import os
import socketserver
import time
import urllib.parse
from search_engine import SearchEngine

PORT = int(os.environ.get("PORT", 8080))
DB_PATH = os.environ.get("DB_PATH", "enwiki-20170820.db")

# Initialize engine
engine = SearchEngine(DB_PATH)
engine.setup()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wikipedia Search Engine | Systems & IR Demo</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: { 50: '#f0f9ff', 500: '#0284c7', 600: '#0369a1', 700: '#075985' }
                    }
                }
            }
        }
    </script>
    <style>
        mark { background-color: #fef08a; color: #854d0e; padding: 0 3px; border-radius: 3px; font-weight: 600; }
        .dark mark { background-color: #ca8a04; color: #fef9c3; }
    </style>
</head>
<body class="bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 min-h-screen transition-colors duration-200">
    <!-- Navbar -->
    <header class="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-sm">W</div>
                <div>
                    <h1 class="font-bold text-base leading-tight">Wikipedia IR Engine</h1>
                    <p class="text-xs text-slate-500 dark:text-slate-400">Disk Inverted Index • BM25 Ranking</p>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <nav class="flex space-x-2 text-sm font-medium">
                    <button id="tab-search-btn" onclick="switchTab('search')" class="px-3 py-1.5 rounded-lg bg-sky-100 dark:bg-sky-950 text-sky-700 dark:text-sky-300">Live Demo</button>
                    <button id="tab-arch-btn" onclick="switchTab('arch')" class="px-3 py-1.5 rounded-lg text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800">Architecture</button>
                </nav>
                <button onclick="toggleDarkMode()" class="p-2 rounded-lg text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800">
                    <span id="theme-icon">🌙</span>
                </button>
            </div>
        </div>
    </header>

    <main class="max-w-4xl mx-auto px-4 py-8">
        <!-- SEARCH TAB -->
        <section id="tab-search">
            <!-- Search Control Box -->
            <div class="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm mb-6">
                <div class="relative">
                    <input 
                        type="text" 
                        id="query-input"
                        placeholder="Search Wikipedia (e.g. 'machine learning', 'search engine', 'quantum')..."
                        value="machine learning"
                        autocomplete="off"
                        class="w-full text-lg pl-12 pr-28 py-3.5 rounded-xl border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 focus:outline-none focus:ring-2 focus:ring-sky-500 dark:focus:ring-sky-400 transition"
                        onkeydown="if(event.key === 'Enter') executeSearch()"
                        oninput="onQueryInput(this.value)"
                    />
                    <span class="absolute left-4 top-4 text-slate-400 text-lg">🔍</span>
                    <button 
                        onclick="executeSearch()"
                        class="absolute right-2 top-2 px-5 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-medium text-sm transition shadow-sm"
                    >Search</button>

                    <!-- Autocomplete Dropdown -->
                    <div id="suggest-box" class="absolute left-0 right-0 top-full mt-1 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-30 hidden overflow-hidden divide-y divide-slate-100 dark:divide-slate-800"></div>
                </div>

                <!-- Algorithm & Stats Bar -->
                <div class="flex flex-wrap items-center justify-between gap-3 mt-4 pt-4 border-t border-slate-100 dark:border-slate-800 text-xs">
                    <div class="flex items-center space-x-2">
                        <span class="text-slate-500 font-medium">Algorithm:</span>
                        <div class="inline-flex rounded-lg border border-slate-200 dark:border-slate-700 p-0.5 bg-slate-100 dark:bg-slate-800">
                            <button id="algo-bm25" onclick="setAlgorithm('bm25')" class="px-2.5 py-1 rounded-md bg-white dark:bg-slate-700 text-sky-600 dark:text-sky-300 font-semibold shadow-xs">Okapi BM25</button>
                            <button id="algo-tfidf" onclick="setAlgorithm('tfidf')" class="px-2.5 py-1 rounded-md text-slate-600 dark:text-slate-400 hover:text-slate-900">TF-IDF</button>
                            <button id="algo-simple" onclick="setAlgorithm('simple')" class="px-2.5 py-1 rounded-md text-slate-600 dark:text-slate-400 hover:text-slate-900">Simple TF</button>
                        </div>
                    </div>
                    <div id="stats-badge" class="text-slate-500 flex items-center space-x-2">
                        <span class="inline-block w-2 h-2 rounded-full bg-emerald-500"></span>
                        <span id="corpus-stats">Loading corpus stats...</span>
                    </div>
                </div>
            </div>

            <!-- Latency / Meta Pill -->
            <div id="search-metrics" class="hidden mb-4 flex items-center justify-between px-2 text-xs text-slate-500 dark:text-slate-400 font-mono">
                <span id="match-count">Found 0 results</span>
                <span id="query-time" class="bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 px-2 py-0.5 rounded border border-emerald-200 dark:border-emerald-800 font-bold">⚡ 0.00 ms</span>
            </div>

            <!-- Results List -->
            <div id="results-container" class="space-y-4"></div>
        </section>

        <!-- ARCHITECTURE TAB -->
        <section id="tab-arch" class="hidden">
            <div class="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-8 shadow-sm space-y-8">
                <div>
                    <h2 class="text-xl font-bold mb-2">Systems Architecture & Binary Storage Layout</h2>
                    <p class="text-sm text-slate-600 dark:text-slate-400">
                        Designed for high throughput and sub-millisecond retrieval without loading a 21GB corpus entirely into RAM.
                    </p>
                </div>

                <!-- Storage layout diagram -->
                <div class="p-5 rounded-xl bg-slate-900 text-slate-200 font-mono text-xs overflow-x-auto border border-slate-800">
                    <div class="text-amber-400 font-bold mb-3">// Binary Postings File Layout: postings.bin</div>
                    <div class="space-y-1">
                        <div>[Offset 0x0000] <span class="text-sky-400">number_of_docs</span> (uint32, 4B)</div>
                        <div class="pl-4 border-l border-slate-700 space-y-1">
                            <div>├── <span class="text-emerald-400">doc_id</span> (uint32, 4B)</div>
                            <div>├── <span class="text-purple-400">term_frequency</span> (uint32, 4B)</div>
                            <div>├── <span class="text-indigo-400">positions_count</span> (uint32, 4B)</div>
                            <div>└── <span class="text-rose-400">positions[]</span> (list of uint32, 4B * tf)</div>
                        </div>
                        <div class="text-slate-500 mt-2">// vocab.pkl stores: {"machine": 0x0000, "learning": 0x0420, ...}</div>
                        <div class="text-slate-500">// Instant seek: file.seek(vocab[token]) -> single I/O read without full scan</div>
                    </div>
                </div>

                <!-- Feature Matrix -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
                        <div class="text-sky-500 font-bold text-base mb-1">Okapi BM25</div>
                        <p class="text-xs text-slate-600 dark:text-slate-400">Probabilistic relevance function with document length normalization and sub-linear term frequency scaling.</p>
                    </div>
                    <div class="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
                        <div class="text-purple-500 font-bold text-base mb-1">Positional Index</div>
                        <p class="text-xs text-slate-600 dark:text-slate-400">Stores exact word offset positions to support phrase search ("machine learning") and 1.5x score boosting.</p>
                    </div>
                    <div class="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
                        <div class="text-emerald-500 font-bold text-base mb-1">O(1) Seek Reads</div>
                        <p class="text-xs text-slate-600 dark:text-slate-400">Vocab index maps tokens directly to byte offsets in binary files, eliminating in-memory posting memory bloat.</p>
                    </div>
                </div>

                <!-- Code embed snippet for portfolio -->
                <div class="border-t border-slate-200 dark:border-slate-800 pt-6">
                    <h3 class="font-bold text-sm mb-2">Embed in Your Portfolio</h3>
                    <p class="text-xs text-slate-500 mb-3">Copy this iframe snippet to show the live interactive demo directly inside your personal website:</p>
                    <div class="p-3 bg-slate-100 dark:bg-slate-800 rounded-lg font-mono text-xs select-all text-slate-700 dark:text-slate-300 overflow-x-auto">
                        &lt;iframe src="http://localhost:8080" width="100%" height="650" style="border:none; border-radius:12px; box-shadow:0 4px 20px rgba(0,0,0,0.08);"&gt;&lt;/iframe&gt;
                    </div>
                </div>
            </div>
        </section>

        <!-- Article Reader Modal -->
        <div id="modal" class="fixed inset-0 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4 z-50 hidden" onclick="closeModal(event)">
            <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 max-w-2xl w-full max-h-[80vh] rounded-2xl p-6 shadow-2xl flex flex-col" onclick="event.stopPropagation()">
                <div class="flex items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800">
                    <h3 id="modal-title" class="font-bold text-lg">Article</h3>
                    <button onclick="document.getElementById('modal').classList.add('hidden')" class="text-slate-400 hover:text-slate-600 text-lg">✕</button>
                </div>
                <div id="modal-body" class="overflow-y-auto py-4 text-sm text-slate-600 dark:text-slate-300 leading-relaxed"></div>
            </div>
        </div>
    </main>

    <script>
        let currentAlgo = 'bm25';

        function toggleDarkMode() {
            document.documentElement.classList.toggle('dark');
            const isDark = document.documentElement.classList.contains('dark');
            document.getElementById('theme-icon').textContent = isDark ? '☀️' : '🌙';
        }

        function switchTab(tab) {
            if (tab === 'search') {
                document.getElementById('tab-search').classList.remove('hidden');
                document.getElementById('tab-arch').classList.add('hidden');
                document.getElementById('tab-search-btn').className = 'px-3 py-1.5 rounded-lg bg-sky-100 dark:bg-sky-950 text-sky-700 dark:text-sky-300 font-semibold';
                document.getElementById('tab-arch-btn').className = 'px-3 py-1.5 rounded-lg text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800';
            } else {
                document.getElementById('tab-search').classList.add('hidden');
                document.getElementById('tab-arch').classList.remove('hidden');
                document.getElementById('tab-arch-btn').className = 'px-3 py-1.5 rounded-lg bg-sky-100 dark:bg-sky-950 text-sky-700 dark:text-sky-300 font-semibold';
                document.getElementById('tab-search-btn').className = 'px-3 py-1.5 rounded-lg text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800';
            }
        }

        function setAlgorithm(algo) {
            currentAlgo = algo;
            ['bm25', 'tfidf', 'simple'].forEach(a => {
                const btn = document.getElementById('algo-' + a);
                if (a === algo) {
                    btn.className = 'px-2.5 py-1 rounded-md bg-white dark:bg-slate-700 text-sky-600 dark:text-sky-300 font-semibold shadow-xs';
                } else {
                    btn.className = 'px-2.5 py-1 rounded-md text-slate-600 dark:text-slate-400 hover:text-slate-900';
                }
            });
            executeSearch();
        }

        async function fetchStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('corpus-stats').textContent = 
                    `${data.doc_count} articles indexed • ${data.vocab_size} vocabulary terms`;
            } catch (e) {
                document.getElementById('corpus-stats').textContent = 'Index active';
            }
        }

        async function executeSearch() {
            const query = document.getElementById('query-input').value.trim();
            if (!query) return;

            const resultsContainer = document.getElementById('results-container');
            resultsContainer.innerHTML = '<div class="text-center py-10 text-slate-400 animate-pulse">Searching index...</div>';

            try {
                const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&algo=${currentAlgo}`);
                const data = await res.json();

                document.getElementById('search-metrics').classList.remove('hidden');
                document.getElementById('match-count').textContent = `Found ${data.results.length} result(s)`;
                document.getElementById('query-time').textContent = `⚡ ${data.latency_ms.toFixed(2)} ms`;

                if (data.results.length === 0) {
                    resultsContainer.innerHTML = `
                        <div class="text-center py-12 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800">
                            <p class="text-slate-500 font-medium">No articles matched "${query}".</p>
                            <p class="text-xs text-slate-400 mt-1">Try keywords like 'machine', 'learning', 'python', 'algorithm'.</p>
                        </div>
                    `;
                    return;
                }

                resultsContainer.innerHTML = data.results.map((item, idx) => `
                    <article class="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-xs hover:border-sky-400 dark:hover:border-sky-600 transition cursor-pointer" onclick="openModal(${item.article_id}, '${escapeHtml(item.title)}')">
                        <div class="flex items-center justify-between mb-2">
                            <div class="flex items-center space-x-2">
                                <span class="text-xs font-bold text-slate-400">#${idx + 1}</span>
                                <h3 class="font-bold text-base text-slate-900 dark:text-slate-100 hover:text-sky-600 transition">${escapeHtml(item.title)}</h3>
                            </div>
                            <span class="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-sky-50 dark:bg-sky-950 text-sky-700 dark:text-sky-300 border border-sky-100 dark:border-sky-800">Score: ${item.score.toFixed(4)}</span>
                        </div>
                        <p class="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">${item.snippet || 'No preview available.'}</p>
                    </article>
                `).join('');
            } catch (err) {
                resultsContainer.innerHTML = `<div class="text-red-500 p-4 bg-red-50 dark:bg-red-950/40 rounded-xl border border-red-200">Error: ${err.message}</div>`;
            }
        }

        async function openModal(id, title) {
            document.getElementById('modal-title').textContent = title;
            document.getElementById('modal-body').textContent = 'Loading article...';
            document.getElementById('modal').classList.remove('hidden');

            try {
                const res = await fetch(`/api/article?id=${id}`);
                const data = await res.json();
                document.getElementById('modal-body').textContent = data.text || 'No text found.';
            } catch (e) {
                document.getElementById('modal-body').textContent = 'Could not load article.';
            }
        }

        function closeModal(event) {
            document.getElementById('modal').classList.add('hidden');
        }

        function escapeHtml(str) {
            return (str || '').replace(/[&<>'"]/g, tag => ({
                '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
            }[tag]));
        }

        let suggestTimeout = null;
        let searchTimeout = null;

        function onQueryInput(val) {
            clearTimeout(suggestTimeout);
            clearTimeout(searchTimeout);
            const box = document.getElementById('suggest-box');
            val = val.trim();

            if (!val) {
                box.classList.add('hidden');
                document.getElementById('results-container').innerHTML = '';
                document.getElementById('search-metrics').classList.add('hidden');
                return;
            }

            // 1. Live typeahead suggestions (100ms)
            suggestTimeout = setTimeout(async () => {
                if (val.length >= 2) {
                    try {
                        const res = await fetch(`/api/suggest?q=${encodeURIComponent(val)}`);
                        const data = await res.json();
                        if (data.suggestions && data.suggestions.length > 0) {
                            box.innerHTML = data.suggestions.map(s => `
                                <div 
                                    class="px-4 py-2 hover:bg-sky-50 dark:hover:bg-slate-800 cursor-pointer text-sm font-medium flex items-center justify-between"
                                    onclick="selectSuggestion('${escapeHtml(s)}')"
                                >
                                    <span>🔍 ${escapeHtml(s)}</span>
                                    <span class="text-xs text-slate-400">suggestion</span>
                                </div>
                            `).join('');
                            box.classList.remove('hidden');
                        } else {
                            box.classList.add('hidden');
                        }
                    } catch (e) {
                        box.classList.add('hidden');
                    }
                } else {
                    box.classList.add('hidden');
                }
            }, 100);

            // 2. Live search execution as user types (250ms debounce)
            searchTimeout = setTimeout(() => {
                executeSearch();
            }, 250);
        }

        function selectSuggestion(word) {
            document.getElementById('query-input').value = word;
            document.getElementById('suggest-box').classList.add('hidden');
            executeSearch();
        }

        document.addEventListener('click', (e) => {
            const box = document.getElementById('suggest-box');
            if (box && !e.target.closest('#query-input') && !e.target.closest('#suggest-box')) {
                box.classList.add('hidden');
            }
        });

        // Initialize on load
        fetchStats();
        executeSearch();
    </script>
</body>
</html>
"""


class SearchEngineRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif path == "/api/stats":
            stats = {
                "doc_count": engine.doc_count,
                "vocab_size": len(engine.vocab),
                "db_path": engine.db_path,
            }
            self._send_json(stats)
            return

        elif path == "/api/suggest":
            prefix = query_params.get("q", [""])[0].strip()
            suggestions = engine.suggest(prefix, limit=6)
            self._send_json({"prefix": prefix, "suggestions": suggestions})
            return

        elif path == "/api/search":
            query = query_params.get("q", [""])[0].strip()
            algo = query_params.get("algo", ["bm25"])[0].lower()

            start_t = time.perf_counter()
            if algo == "tfidf":
                raw_results = engine.search_tfidf(query)
            elif algo == "simple":
                raw_results = engine.search_simple(query)
            else:
                raw_results = engine.search_bm25(query)
            elapsed_ms = (time.perf_counter() - start_t) * 1000

            query_tokens = engine._process(query)
            enriched_results = []
            for article_id, score in raw_results:
                title, text = engine.get_article_text(article_id)
                snippet = self._generate_html_snippet(text, query_tokens)
                enriched_results.append(
                    {
                        "article_id": article_id,
                        "title": title,
                        "score": score,
                        "snippet": snippet,
                    }
                )

            response = {
                "query": query,
                "algo": algo,
                "latency_ms": elapsed_ms,
                "results": enriched_results,
            }
            self._send_json(response)
            return

        elif path == "/api/article":
            art_id = query_params.get("id", [None])[0]
            if art_id is not None:
                try:
                    title, text = engine.get_article_text(int(art_id))
                    self._send_json({"article_id": int(art_id), "title": title, "text": text})
                    return
                except ValueError:
                    pass
            self._send_json({"error": "Article not found"}, status=404)
            return

        else:
            self.send_response(404)
            self.end_headers()

    def _generate_html_snippet(self, text, query_tokens, max_words=32):
        if not text:
            return ""
        words = text.split()
        if not words:
            return ""

        lower_words = [w.lower().strip(".,!?;:\"'()[]") for w in words]
        query_set = set(query_tokens)

        best_idx = 0
        for idx, word in enumerate(lower_words):
            if word in query_set:
                best_idx = idx
                break

        start = max(0, best_idx - 8)
        end = min(len(words), start + max_words)
        snippet_words = words[start:end]

        formatted = []
        for word in snippet_words:
            cleaned = word.lower().strip(".,!?;:\"'()[]")
            if cleaned in query_set:
                formatted.append(f"<mark>{html_escape(word)}</mark>")
            else:
                formatted.append(html_escape(word))

        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(words) else ""
        return f"{prefix} {' '.join(formatted)} {suffix}".strip()

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def html_escape(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def run_server():
    print(f"🚀 Starting Wikipedia Search Engine server on http://localhost:{PORT}")
    print(f"📁 Database: {DB_PATH}")
    print(f"⚡ Press Ctrl+C to stop.")
    with socketserver.TCPServer(("", PORT), SearchEngineRequestHandler) as httpd:
        httpd.allow_reuse_address = True
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully...")


if __name__ == "__main__":
    run_server()

