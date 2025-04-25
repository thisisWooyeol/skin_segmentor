import gradio as gr

if __name__ == "__main__":
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="stone",
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600)), "arial"],
    )

    with gr.Blocks(theme=theme) as demo:
        with gr.Column(elem_classes="header"):
            gr.Markdown("# 🔍 Aramhuvis x SNU: Anomaly Detect Skin Disease")
            gr.Markdown("### Wooyeol Lee, Minseo Kim, Byeongho Park")
            gr.Markdown("[[GitHub](https://github.com/thisiswooyeol/skin_segmentor)]")

        with gr.Column(elem_classes="abstract"):
            gr.Markdown(
                "MESA is a novel generative model based on latent denoising diffusion capable of generating 2.5D representations of terrain based on the text prompt conditioning supplied via natural language. The model produces two co-registered modalities of optical and depth maps."
            )  # Replace with your abstract text
            gr.Markdown(
                "This is a test version of the demo app. Please be aware that MESA supports primarily complex, mountainous terrains as opposed to flat land"
            )
            gr.Markdown(
                "> ⚠️ **The generated image is quite large, so for the larger resolution (768) it might take a while to load the surface**"
            )
            with gr.Row():
                prompt_input = gr.Textbox(
                    lines=2, placeholder="Enter a terrain description..."
                )

            with gr.Tabs() as output_tabs:
                with gr.Tab("2D View (Fast)"):
                    generate_2d_button = gr.Button(
                        "Generate Terrain", variant="primary"
                    )
                    with gr.Row():
                        rgb_output = gr.Image(label="RGB Image")
                        elevation_output = gr.Image(label="Elevation Map")

                with gr.Tab("3D View (Slow)"):
                    generate_3d_button = gr.Button(
                        "Generate Terrain", variant="primary"
                    )
                    model_3d_output = gr.Model3D(
                        camera_position=[90, 135, 512],
                        clear_color=[0.0, 0.0, 0.0, 0.0],
                        # display_mode = 'point_cloud'
                    )

            with gr.Accordion("Advanced Options", open=False) as advanced_options:
                num_inference_steps_slider = gr.Slider(
                    minimum=10, maximum=1000, step=10, value=50, label="Inference Steps"
                )
                guidance_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=7.5,
                    label="Guidance Scale",
                )
                seed_number = gr.Number(value=6378, label="Seed")
                random_seed = gr.Checkbox(value=True, label="Random Seed")
                crop_size_slider = gr.Slider(
                    minimum=128,
                    maximum=768,
                    step=64,
                    value=768,
                    label="(3D Only) Crop Size",
                )
                vertex_count_slider = gr.Slider(
                    minimum=0,
                    maximum=10000,
                    step=0,
                    value=0,
                    label="(3D Only) Vertex Count (Default: 0 - no reduction)",
                )
                prefix_textbox = gr.Textbox(
                    label="Prompt Prefix", value="A Sentinel-2 image of "
                )

            demo.queue().launch()
