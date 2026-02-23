# 🧬 ResearchMind — Autonomous AI Scientist

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/Vi-bha/ResearchMind)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vi-bha/ResearchMind/blob/main/ResearchMind.ipynb)

> Autonomous AI system that fetches research papers from PubMed, performs RAG-based analysis, generates novel hypotheses, designs experiments and produces publication-style research proposals — with zero human intervention.

## 🚀 Autonomous Pipeline
```
Topic Input
    → 🔍 PubMed Fetch (35M+ papers)
        → 🧠 RAG Analysis (FAISS + SentenceTransformers)
            → 💡 Hypothesis Generation (Groq LLaMA 3.1)
                → 🧪 Experiment Design
                    → 📝 Research Proposal Output
```

## ✨ Features

| Step | Feature | Description |
|------|---------|-------------|
| 1 | PubMed Search | Auto-fetches top papers for any topic |
| 2 | Literature Analysis | RAG pipeline summarizes key findings |
| 3 | Hypothesis Generation | LLM generates novel, testable hypotheses |
| 4 | Experiment Design | Dataset, methodology, timeline, controls |
| 5 | Research Proposal | Publication-style output with peer review |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Paper Source | PubMed API (35M+ papers, no rate limit) |
| Vector Store | FAISS |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| LLM | Groq LLaMA 3.1 (8B Instant) |
| UI | Gradio |

## ⚙️ How to Run
```bash
# Install dependencies
pip install faiss-cpu sentence-transformers groq gradio requests

# Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY="your_key_here"

# Run
python app.py
```

Or open directly in Google Colab using the badge above.

## 📦 Requirements
```
groq
faiss-cpu
sentence-transformers
gradio
requests
```

## 💡 Example Output

**Input topic:** `large language models medical imaging`

**Output includes:**
- 5 fetched papers from PubMed (2024–2025)
- RAG-based literature summary
- 3 novel hypotheses
- Full experiment design with methodology
- Publication-style research proposal

## 🔗 Links
- 🤗 Live Demo: [huggingface.co/spaces/Vi-bha/ResearchMind](https://huggingface.co/spaces/Vi-bha/ResearchMind)
- 🔍 PaperLens: [huggingface.co/spaces/Vi-bha/PaperLens](https://huggingface.co/spaces/Vi-bha/PaperLens)
- 🔬 MedLens: [huggingface.co/spaces/Vi-bha/MedLens](https://huggingface.co/spaces/Vi-bha/MedLens)

---
*Built by Vibhavari Tummewar | MTech Advanced Computing, MANIT Bhopal*
