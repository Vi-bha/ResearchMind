# 🧬 ResearchMind — Autonomous AI Scientist

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Demo-Live-blue)](https://huggingface.co/spaces/Vi-bha/ResearchMind)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Autonomous 5-stage AI research pipeline that queries PubMed's 35M+ papers,
performs RAG-based literature analysis, generates novel hypotheses, designs
experiments, and produces publication-style research proposals — with zero
human intervention.

---

## Pipeline

```
Topic Input
  → Stage 1: PubMed Fetch      (35M+ papers, structured metadata)
  → Stage 2: RAG Indexing      (FAISS + all-MiniLM-L6-v2 embeddings)
  → Stage 3: Hypothesis Gen    (Groq LLaMA 3.1, top-3 semantic retrieval)
  → Stage 4: Experiment Design (dataset, methodology, metrics, timeline)
  → Stage 5: Research Report   (publication-style proposal + peer critique)
```

## Folder Structure

```
ResearchMind/
├── app.py            ← Gradio UI entry point
├── pipeline.py       ← Core autonomous research logic
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/Vi-bha/ResearchMind
cd ResearchMind
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here   # free at console.groq.com
python app.py
```

Open `http://localhost:7860` in your browser.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Paper Source | PubMed API (35M+ papers) |
| Vector Store | FAISS IndexFlatL2 |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 (384-dim) |
| LLM | Groq LLaMA 3.1 8B Instant |
| UI | Gradio 4.x |

## Example Output

**Input:** `large language models medical imaging`

**Output includes:**
- 5 fetched PubMed papers (2023–2025) with metadata
- RAG-based literature summary with identified research gaps
- Novel, testable hypothesis with rationale
- Experiment design with dataset, architecture, metrics, timeline
- Peer critique with feasibility score
- Full publication-style research proposal (~1,500 words)

## Related Projects

| Project | Description | Demo |
|---------|-------------|------|
| [PaperLens](https://github.com/Vi-bha/PaperLens) | RAG pipeline for research paper Q&A | [🤗 Live](https://huggingface.co/spaces/Vi-bha/PaperLens) |
| [MedLens](https://github.com/Vi-bha/MedLens) | Multimodal AI for prostate MRI analysis | [🤗 Live](https://huggingface.co/spaces/Vi-bha/MedLens) |

## Author

**Vibhavari Tummewar** — M.Tech Advanced Computing, MANIT Bhopal
Scopus-indexed researcher in LLM-assisted medical AI systems.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/vibhavari-tummewar)
[![HuggingFace](https://img.shields.io/badge/🤗-Vi--bha-yellow)](https://huggingface.co/Vi-bha)
