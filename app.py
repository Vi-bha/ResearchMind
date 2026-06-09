"""
ResearchMind — Gradio UI
Run: python app.py
"""

import os
import gradio as gr
from pipeline import ResearchMindPipeline

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not set. Export it: export GROQ_API_KEY=your_key"
    )

pipeline = ResearchMindPipeline(groq_api_key=GROQ_API_KEY)


def run_researchmind(topic: str, max_papers: int):
    if not topic.strip():
        return "❌ Please enter a research topic.", "", "", ""

    try:
        papers = pipeline.fetch_papers(topic, max_papers=int(max_papers))
        if not papers:
            return "❌ No papers found. Try a different topic.", "", "", ""

        pipeline.build_knowledge_base(papers)
        results = pipeline.run(topic, papers)

        papers_md = "## 📚 Papers Analyzed\n\n"
        for i, p in enumerate(papers):
            papers_md += (
                f"**[{i+1}] {p['title']}**\n"
                f"_{', '.join(p['authors'])} — {p['journal']} ({p['year']})_\n"
                f"[PubMed ↗]({p['url']})\n\n"
            )

        return (
            papers_md,
            results["summary"],
            results["hypothesis"] + "\n\n---\n\n" + results["experiment"],
            results["report"],
        )

    except Exception as e:
        return f"❌ Error: {e}", "", "", ""


with gr.Blocks(title="ResearchMind — Autonomous AI Scientist",
               theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧬 ResearchMind — Autonomous AI Scientist
    Enter any research topic → ResearchMind autonomously fetches papers from PubMed,
    analyzes findings via RAG, generates hypotheses, designs experiments,
    and produces a full publication-style research proposal.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            topic_input = gr.Textbox(
                label="Research Topic",
                placeholder="e.g. large language models medical imaging",
            )
        with gr.Column(scale=1):
            papers_slider = gr.Slider(
                minimum=3, maximum=10, value=5, step=1,
                label="Number of Papers",
            )

    run_btn = gr.Button("🚀 Run ResearchMind", variant="primary", size="lg")

    with gr.Tabs():
        with gr.Tab("📚 Papers"):
            papers_output = gr.Markdown()
        with gr.Tab("📊 Literature Summary"):
            summary_output = gr.Markdown()
        with gr.Tab("💡 Hypothesis + Experiment"):
            hypothesis_output = gr.Markdown()
        with gr.Tab("📄 Full Research Report"):
            report_output = gr.Markdown()

    run_btn.click(
        fn=run_researchmind,
        inputs=[topic_input, papers_slider],
        outputs=[papers_output, summary_output,
                 hypothesis_output, report_output],
    )

if __name__ == "__main__":
    demo.launch()
