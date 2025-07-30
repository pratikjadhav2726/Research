import os
from typing import List

try:
    import openai
except ImportError:
    openai = None

from .config import GENERATOR_BACKEND, TOP_K_CANDIDATES
from .retriever import HierarchicalRetriever
from .index import HierarchicalIndex, Node

_SYSTEM_PROMPT = """You are an intelligent retrieval agent responsible for selecting the right abstraction level in a hierarchical knowledge index. Given a user query and short description of each level, output the level number (0 for most detailed, higher numbers for more abstract) that should be searched first to best answer the query. Reply with only the level number (integer)."""


class RetrievalAgent:
    """Agentic controller that decides retrieval level and iteratively refines context."""

    def __init__(self, index_name: str, max_iterations: int = 3):
        self.index_name = index_name
        self.retriever = HierarchicalRetriever(index_name)
        self.index = self.retriever.index
        self.max_iterations = max_iterations

        # Build mapping of level -> brief description (summary of root at that level)
        self._level_desc = self._compute_level_descriptions()

    # ---------- Public API -----------

    def answer(self, query: str) -> str:
        # 1. Decide abstraction level
        level = self._decide_level(query)
        # 2. Iterative retrieve-critique loop
        visited_levels = set()
        context_nodes: List[Node] = []
        for _ in range(self.max_iterations):
            visited_levels.add(level)
            retrieved = self.retriever.search(query, level=level, top_k=TOP_K_CANDIDATES)
            context_nodes = [n for n, _ in retrieved]
            # Evaluate sufficiency via heuristic (length + presence of keywords)
            if self._context_sufficient(query, context_nodes):
                break
            # Decide to zoom in or out if not sufficient
            level = self._refine_level(query, current_level=level, visited_levels=visited_levels)
            if level is None:
                break  # cannot improve further
        # 3. Generate final answer
        answer = self._generate_answer(query, context_nodes)
        return answer

    # ---------- Internal helpers ---------

    def _decide_level(self, query: str) -> int:
        if GENERATOR_BACKEND == "openai" and openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            prompt = _SYSTEM_PROMPT + "\nUser query: " + query + "\n" + "\n".join(
                f"Level {lvl}: {desc}" for lvl, desc in self._level_desc.items()
            )
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            )
            try:
                lvl_int = int(response.choices[0].message.content.strip())
                return lvl_int
            except Exception:
                pass  # fallback
        # Fallback heuristic: if query length <= 5 words => summary level else detailed
        return 1 if len(query.split()) <= 5 else 0

    def _compute_level_descriptions(self):
        level_desc = {}
        for node in self.index.nodes.values():
            # pick first node encountered at each level as descriptor
            if node.level not in level_desc:
                level_desc[node.level] = node.text[:80].replace("\n", " ") + "..."
        return level_desc

    def _context_sufficient(self, query: str, nodes: List[Node]) -> bool:
        # simple heuristic: if combined text length exceeds 400 words, consider sufficient
        total_words = sum(len(n.text.split()) for n in nodes)
        return total_words >= 400 or len(nodes) >= TOP_K_CANDIDATES

    def _refine_level(self, query: str, current_level: int, visited_levels: set) -> int:
        # naive strategy: if current level is abstract (>0), zoom in; else zoom out.
        next_level = current_level - 1 if current_level > 0 else current_level + 1
        if next_level in visited_levels or next_level < 0:
            return None
        return next_level

    def _generate_answer(self, query: str, context_nodes: List[Node]) -> str:
        context_text = "\n".join(n.text for n in context_nodes)
        if GENERATOR_BACKEND == "openai" and openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            prompt = (
                "You are a helpful assistant. Use the following context to answer the question. "
                "Cite relevant evidence snippets when appropriate.\n\n"
                f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
            )
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613", messages=[{"role": "user", "content": prompt}], temperature=0.2
            )
            return response.choices[0].message.content.strip()
        else:
            # Fallback: simple template answer
            return f"[Mock Answer] {query}\n\nRelevant context:\n{context_text[:500]}..."