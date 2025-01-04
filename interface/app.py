"""Main application for the MarqetSim interface."""

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# MarqetSim: A simulation based for product discovery.")

if __name__ == "__main__":
    demo.launch()
