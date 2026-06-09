"""
ResearchMind — Autonomous Research Pipeline
Handles PubMed fetching, RAG knowledge base, and 5-stage autonomous analysis.
"""

import os
import json
import time
import xml.etree.ElementTree as ET

import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


class ResearchMindPipeline:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    # ─────────────────────────────────────────────
    # Stage 1 — PubMed Fetch
    # ─────────────────────────────────────────────
    def fetch_papers(self, topic: str, max_papers: int = 5) -> list[dict]:
        """Search PubMed and return structured paper metadata."""
        print(f"🔍 Searching PubMed for: '{topic}'")

        ids = self._search_pubmed_ids(topic, max_papers)
        if not ids:
            print("⚠️  No papers found.")
            return []

        papers = []
        for pmid in ids:
            time.sleep(0.5)
            paper = self._fetch_paper_details(pmid)
            if paper:
                papers.append(paper)
                print(f"  📄 {paper['title'][:70]}...")

        print(f"✅ Fetched {len(papers)} papers\n")
        return papers

    def _search_pubmed_ids(self, topic: str, max_papers: int) -> list[str]:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": topic,
                    "retmax": max_papers, "retmode": "json", "sort": "relevance"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["esearchresult"]["idlist"]

    def _fetch_paper_details(self, pmid: str) -> dict | None:
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": pmid,
                        "retmode": "xml", "rettype": "abstract"},
                timeout=10,
            )
            root = ET.fromstring(r.content)
            return {
                "pmid":     pmid,
                "title":    root.findtext(".//ArticleTitle") or "N/A",
                "abstract": root.findtext(".//AbstractText") or "No abstract available.",
                "year":     root.findtext(".//PubDate/Year") or "N/A",
                "journal":  root.findtext(".//Journal/Title") or "N/A",
                "authors": [
                    f"{a.findtext('LastName')} {a.findtext('ForeName', '')}"
                    for a in root.findall(".//Author")[:3]
                ],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        except Exception as e:
            print(f"  ⚠️  Skipped PMID {pmid}: {e}")
            return None

    # ─────────────────────────────────────────────
    # Stage 2 — RAG Knowledge Base
    # ─────────────────────────────────────────────
    def build_knowledge_base(self, papers: list[dict]) -> None:
        """Embed paper abstracts and build FAISS index."""
        print("🧠 Building knowledge base...")
        self.chunks = [self._paper_to_chunk(p) for p in papers]
        embeddings = self.embedder.encode(
            self.chunks, show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"✅ {self.index.ntotal} papers indexed\n")

    def _paper_to_chunk(self, p: dict) -> str:
        return (
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Journal: {p['journal']} ({p['year']})\n"
            f"Abstract: {p['abstract']}\n"
            f"URL: {p['url']}"
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return top-k relevant chunks for a query."""
        if self.index is None:
            raise RuntimeError("Knowledge base not built. Call build_knowledge_base() first.")
        q_emb = self.embedder.encode([query]).astype("float32")
        _, idxs = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in idxs[0]]

    # ─────────────────────────────────────────────
    # LLM Helper
    # ─────────────────────────────────────────────
    def _llm(self, prompt: str, max_tokens: int = 1500) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    # ─────────────────────────────────────────────
    # Stages 3–7 — Autonomous Research Loop
    # ─────────────────────────────────────────────
    def run(self, topic: str, papers: list[dict]) -> dict:
        """
        Execute all 5 autonomous research stages and return results dict.
        Assumes build_knowledge_base() has already been called.
        """
        context = "\n\n---\n\n".join(self.retrieve(topic, top_k=3))

        stages = {
            "summary":    self._summarize(context),
            "hypothesis": None,
            "experiment": None,
            "critique":   None,
            "report":     None,
        }

        stages["hypothesis"] = self._hypothesize(topic, stages["summary"])
        stages["experiment"] = self._design_experiment(stages["hypothesis"])
        stages["critique"]   = self._critique(stages["experiment"])
        stages["report"]     = self._compile_report(topic, stages)

        return stages

    def _summarize(self, context: str) -> str:
        print("  Step 1/5 — Summarizing literature...")
        return self._llm(f"""You are ResearchMind, an autonomous AI scientist.

Based on these research papers:
{context}

Provide a structured literature summary:
1. KEY FINDINGS — What do these papers collectively show?
2. CURRENT STATE — Where is the field right now?
3. RESEARCH GAPS — What is still unknown or unsolved?

Be concise and scientific.""")

    def _hypothesize(self, topic: str, summary: str) -> str:
        print("  Step 2/5 — Generating hypothesis...")
        return self._llm(f"""You are ResearchMind, an autonomous AI scientist.

Topic: {topic}
Literature Summary:
{summary}

Generate a novel, testable research hypothesis:
1. HYPOTHESIS — One clear, specific statement
2. RATIONALE — Why this is worth testing
3. NOVELTY — How it goes beyond current research
4. EXPECTED OUTCOME — What would confirm it

Be bold and scientifically rigorous.""")

    def _design_experiment(self, hypothesis: str) -> str:
        print("  Step 3/5 — Designing experiment...")
        return self._llm(f"""You are ResearchMind, an autonomous AI scientist.

Hypothesis:
{hypothesis}

Design a concrete experiment:
1. DATASET — What data is needed and where to get it
2. MODEL ARCHITECTURE — What AI/ML approach to use
3. METHODOLOGY — Step-by-step procedure
4. EVALUATION METRICS — How to measure success
5. EXPECTED TIMELINE — Realistic schedule

Be specific and implementable.""")

    def _critique(self, experiment: str) -> str:
        print("  Step 4/5 — Peer reviewing...")
        return self._llm(f"""You are ResearchMind acting as a critical peer reviewer.

Experiment Plan:
{experiment}

Critically evaluate:
1. STRENGTHS — What is well designed?
2. WEAKNESSES — What could fail?
3. RISKS — Biggest risks?
4. IMPROVEMENTS — How to strengthen?
5. FEASIBILITY SCORE — Rate 1–10 with justification

Be harsh but constructive.""")

    def _compile_report(self, topic: str, stages: dict) -> str:
        print("  Step 5/5 — Compiling report...")
        return self._llm(f"""You are ResearchMind, an autonomous AI scientist.

Compile a publication-style research proposal for: {topic}

LITERATURE SUMMARY:
{stages['summary']}

HYPOTHESIS:
{stages['hypothesis']}

EXPERIMENT DESIGN:
{stages['experiment']}

PEER REVIEW:
{stages['critique']}

Include: Abstract (150 words), Introduction, Research Gap,
Proposed Hypothesis, Methodology, Expected Impact, Conclusion.
Format professionally.""", max_tokens=2000)
