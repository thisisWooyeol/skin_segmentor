# annotation_gui.py

import argparse
import os

import gradio as gr
import numpy as np
from PIL import Image


def list_images(image_folder):
    extensions = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(extensions)]
    )


def load_and_prepare(file_name, image_folder):
    path = os.path.join(image_folder, file_name)
    image = Image.open(path).convert("RGB")
    array = np.array(image)
    return gr.update(value=array)


def save_mask(file_name, editor_value, image_folder):
    if not isinstance(editor_value, dict):
        return "No drawing data available to save."

    layers = editor_value.get("layers", [])

    if layers:
        first = layers[0]
        if isinstance(first, np.ndarray):
            h, w = first.shape[:2]
        elif isinstance(first, Image.Image):
            w, h = first.size
        else:
            return "Unsupported layer data format."
    else:
        bg = editor_value.get("background")
        if isinstance(bg, np.ndarray):
            h, w = bg.shape[:2]
        elif isinstance(bg, Image.Image):
            w, h = bg.size
        else:
            return "No mask data to save."

    mask = np.zeros((h, w), dtype=np.uint8)
    for layer in layers:
        if isinstance(layer, np.ndarray):
            if layer.ndim == 3 and layer.shape[2] == 4:
                alpha = layer[:, :, 3]
            else:
                alpha = np.any(layer != 0, axis=-1)
            mask[alpha > 0] = 255
        else:
            gray = np.array(layer.convert("L"))
            mask[gray > 0] = 255

    label_dir = os.path.normpath(os.path.join(image_folder, os.pardir, "label"))
    os.makedirs(label_dir, exist_ok=True)

    mask_img = Image.fromarray(mask, mode="L")
    base, _ = os.path.splitext(file_name)
    mask_filename = f"{base}_mask.png"
    save_path = os.path.join(label_dir, mask_filename)
    mask_img.save(save_path)
    return f"Mask saved to: {save_path}"


def prev_next_image(current_idx, direction, files):
    """
    Returns the new index and filename after moving prev/next.
    direction: -1 for prev, +1 for next
    """
    num_files = len(files)
    new_idx = current_idx + direction
    new_idx = max(0, min(new_idx, num_files - 1))  # Clamp to valid range
    return new_idx, files[new_idx]


def build_interface(image_folder):
    files = list_images(image_folder)
    if not files:
        raise ValueError("No images found in the provided folder.")

    with gr.Blocks() as demo:
        gr.Markdown("# Mask Annotation GUI\nSelect an image and draw your mask.")

        with gr.Row():
            prev_btn = gr.Button("⬅️ Prev", elem_id="prev-btn")
            dropdown = gr.Dropdown(choices=files, label="Select Image", value=files[0])
            next_btn = gr.Button("Next ➡️", elem_id="next-btn")

        # Track current image index in a hidden state
        idx_state = gr.State(0)

        mask_draw = gr.ImageEditor(
            type="numpy",
            label="Draw Mask",
            brush=gr.Brush(colors=["#FFFFFF80"], color_mode="fixed"),
            interactive=True,
        )

        save_button = gr.Button("Save Mask")
        save_output = gr.Textbox(label="Save Status")

        # Load image when dropdown changes
        dropdown.change(
            fn=lambda file_name, image_folder: load_and_prepare(
                file_name, image_folder
            ),
            inputs=[dropdown, gr.State(image_folder)],
            outputs=[mask_draw],
        )

        # Save mask
        save_button.click(
            fn=save_mask,
            inputs=[dropdown, mask_draw, gr.State(image_folder)],
            outputs=[save_output],
        )

        # Prev button logic
        def on_prev(idx, files, image_folder):
            new_idx, new_file = prev_next_image(idx, -1, files)
            image = load_and_prepare(new_file, image_folder)
            return new_idx, gr.update(value=new_file), image

        prev_btn.click(
            fn=on_prev,
            inputs=[idx_state, gr.State(files), gr.State(image_folder)],
            outputs=[idx_state, dropdown, mask_draw],
        )

        # Next button logic
        def on_next(idx, files, image_folder):
            new_idx, new_file = prev_next_image(idx, +1, files)
            image = load_and_prepare(new_file, image_folder)
            return new_idx, gr.update(value=new_file), image

        next_btn.click(
            fn=on_next,
            inputs=[idx_state, gr.State(files), gr.State(image_folder)],
            outputs=[idx_state, dropdown, mask_draw],
        )

        # Sync dropdown selection to index state
        def dropdown_to_idx(selected_file, files):
            idx = files.index(selected_file)
            return idx

        dropdown.change(
            fn=dropdown_to_idx,
            inputs=[dropdown, gr.State(files)],
            outputs=[idx_state],
        )

        # Auto-load first image on startup
        demo.load(
            fn=load_and_prepare,
            inputs=[gr.State(files[0]), gr.State(image_folder)],
            outputs=[mask_draw],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask Annotation GUI using Gradio")
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to the directory containing images to annotate",
    )
    args = parser.parse_args()
    app = build_interface(args.image_folder)
    app.launch()
