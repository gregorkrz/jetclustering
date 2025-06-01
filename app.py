import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import os

from src.model_wrapper_gradio import inference

# === Dummy file-based prefill function ===
def prefill_event(subdataset, event_idx):
    base_path = f"demo_datasets/{subdataset}/{event_idx}"
    try:
        with open(f"{base_path}.txt", "r") as f:
            particles_data = f.read()
    except FileNotFoundError:
        particles_data = "pt eta phi mass charge\n"

    try:
        with open(f"{base_path}_quarks.txt", "r") as f:
            quarks_data = f.read()
    except FileNotFoundError:
        quarks_data = "pt eta phi\n"

    return particles_data, quarks_data


#from huggingface_hub import snapshot_download
#snapshot_download(repo_id="gregorkrzmanc/jetclustering", local_dir="models/")
#snapshot_download(repo_id="gregorkrzmanc/jetclustering_demo", local_dir="demo_datasets/", repo_type="dataset")

# === Interface layout ===
def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Jet Clustering Demo")
        # now put a short text explaining that the demo is very slow and if you want to run it on your machine, you can use the following docker-compose file:
        # version: '3.8'
        #
        # services:
        #   jetclustering_demo:
        #     image: gkrz/jetclustering_demo_cpu:v0
        #     ports:
        #       - "7860:7860"
        gr.Markdown("The live demo is very slow (usually takes 1-5 minutes for a single event). If you want to run it on your machine, you can use the following docker-compose file:\n\n```yaml\nversion: '3.8'\n\nservices:\n  jetclustering_demo:\n    image: gkrz/jetclustering_demo_cpu:v0\n    ports:\n      - '7860:7860'\n```")
        with gr.Row():
            loss_dropdown = gr.Dropdown(choices=["GP_IRC_SN", "GP_IRC_S", "GP", "base"], label="Loss Function", value="GP_IRC_SN")
            train_dataset_dropdown = gr.Dropdown(choices=["QCD", "900_03", "900_03+700_07", "700_07", "900_03+700_07+QCD"], label="Training Dataset", value="QCD")

        with gr.Row():
            subdataset_dropdown = gr.Dropdown(choices=[x for x in os.listdir("demo_datasets") if not x.startswith(".")], label="Subdataset", value="QCD")
            event_idx_dropdown = gr.Dropdown(choices=list(range(20)), label="Event Index", value=15)
        prefill_btn = gr.Button("Load Event from Dataset")

        particles_text = gr.Textbox(label="Particles CSV (pt eta phi mass charge)", lines=6, interactive=True)
        quarks_text = gr.Textbox(label="Quarks CSV (pt eta phi) - optional", lines=3, interactive=True)

        process_btn = gr.Button("Run Jet Clustering")

        image_output = gr.Plot(label="Output")
        model_jets_output = gr.JSON(label="Model Jets")
        antikt_jets_output = gr.JSON(label="Anti-kt Jets")

        prefill_btn.click(fn=prefill_event,
                          inputs=[subdataset_dropdown, event_idx_dropdown],
                          outputs=[particles_text, quarks_text])


        process_btn.click(fn=inference,
                          inputs=[loss_dropdown, train_dataset_dropdown, particles_text, quarks_text],
                          outputs=[model_jets_output, antikt_jets_output, image_output])

    return demo


demo = gradio_ui()
demo.launch()

