import gradio as gr
import numpy as np
import mlxu
import os
import re
import torch

from io import BytesIO
from natsort import natsorted
from PIL import Image

from inference import LocalInferenceModel

FLAGS, _ = mlxu.define_flags_with_default(
    host='0.0.0.0',
    port=5000,
    dtype='float16',
    checkpoint='Emma02/LVM_ckpts',
    torch_devices='',
    context_frames=16,
)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_example_image_groups(directory):
    example_groups = {}
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            example_groups[subdir] = []
            images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            images = natsorted(images, key=natural_sort_key)
            for filename in images:
                img = Image.open(os.path.join(subdir_path, filename))
                example_groups[subdir].append(img)
    return example_groups

def main(_):
    assert FLAGS.checkpoint != ''

    model = LocalInferenceModel(
        checkpoint=FLAGS.checkpoint,
        torch_device=torch.device("cuda"),
        dtype=FLAGS.dtype,
        context_frames=FLAGS.context_frames,
        use_lock=False,
    )

    checkerboard_r1 = np.concatenate([np.zeros((8, 8, 3)), np.ones((8, 8, 3)), np.zeros((8, 8, 3))], axis=1)
    checkerboard_r2 = np.concatenate([np.ones((8, 8, 3)), np.zeros((8, 8, 3)), np.ones((8, 8, 3))], axis=1)
    checkerboard = np.concatenate([checkerboard_r1, checkerboard_r2] * 16, axis=0).astype(np.float32)

    def generate_images(input_images, n_new_frames, n_candidates, temperature=1.0, top_p=0.9):
        assert len(input_images) > 0
        input_images = [
            np.array(img.convert('RGB').resize((256, 256)), dtype=np.float32) / 255.0
            for img in input_images
        ]
        input_images = np.stack(input_images, axis=0)
        output_images = model([input_images], n_new_frames, n_candidates, temperature, top_p)[0]

        generated_images = []
        for candidate in output_images:
            concatenated_image = []
            for i, img in enumerate(candidate):
                concatenated_image.append(img)
                if i < len(candidate) - 1:
                    concatenated_image.append(checkerboard)
            generated_images.append(
                Image.fromarray(
                    (np.concatenate(concatenated_image, axis=1) * 255).astype(np.uint8)
                )
            )

        return generated_images

    with gr.Blocks(css="""
        .small-button {
            padding: 5px 10px;
            min-width: 80px;
        }
        .large-gallery img {
            width: 100%;
            height: auto;
            max-height: 150px;
        }
    """) as demo:
        with gr.Column():
            image_list = gr.State([])
            gr.Markdown('# LVM Demo')
            gr.Markdown('This is the demo of CVPR 2024 paper: Sequential Modeling Enables Scalable Learning for Large Vision Models. For more information about this paper please check the [website](https://yutongbai.com/lvm.html).')
            gr.Markdown(f'Serving model: [{FLAGS.checkpoint}](https://huggingface.co/{FLAGS.checkpoint})')
            
            gr.Markdown('**There are mainly two visual prompting: sequential prompting and analogy prompting.**')
            gr.Markdown('**For analogy prompting: describe the task with few-shot examples, which is pairs of (x, y) inputs where x is the input image and y the "annotated" image. And add one query image in the end. Download the few-shot examples dataset at [this link](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/ybai20_jh_edu/Ei0xiLdFFqJPnwAlFWar29EBUAvB0O3CVaJykZl-f11KDQ?e=Bx9SXZ), and you can simply change the query image in the end for testing.**')
            gr.Markdown('**For sequential prompting, input a sequence of continuous frames and let the model generate the next one. Please refer to the default examples below.**')
            gr.Markdown('## Inputs')
            with gr.Row():
                upload_drag = gr.File(
                    type='binary',
                    file_types=['image'],
                    file_count='multiple',
                )
                with gr.Column():
                    gen_length_slider = gr.Slider(
                        label='Generation length',
                        minimum=1,
                        maximum=32,
                        value=1,
                        step=1,
                        interactive=True,
                    )
                    n_candidates_slider = gr.Slider(
                        label='Number of candidates',
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1,
                        interactive=True,
                    )
                    temp_slider = gr.Slider(
                        label='Temperature',
                        minimum=0,
                        maximum=2.0,
                        value=1.0,
                        interactive=True,
                    )
                    top_p_slider = gr.Slider(
                        label='Top p',
                        minimum=0,
                        maximum=1.0,
                        value=0.9,
                        interactive=True,
                    )
                    clear_btn = gr.Button(
                        value='Clear',
                        elem_classes=['small-button'],
                    )
                    generate_btn = gr.Button(
                        value='Generate',
                        interactive=False,
                        elem_classes=['small-button'],
                    )
            input_gallery = gr.Gallery(
                columns=7,
                rows=1,
                object_fit='scale-down',
                label="Input image sequence"
            )
            gr.Markdown('## Outputs (multi candidates)')
            output_gallery = gr.Gallery(
                columns=4,
                object_fit='scale-down',
                label="Output image"
            )

        def upload_image_fn(files, images):
            for file in files:
                images.append(Image.open(BytesIO(file)))

            return {
                upload_drag: None,
                image_list: images,
                input_gallery: images,
                generate_btn: gr.update(interactive=True),
            }

        def clear_fn():
            return {
                image_list: [],
                input_gallery: [],
                generate_btn: gr.update(interactive=False),
                output_gallery: [],
            }

        def disable_generate_btn():
            return {
                generate_btn: gr.update(interactive=False),
            }

        def generate_fn(images, n_candidates, gen_length, temperature, top_p):
            new_images = generate_images(
                images,
                gen_length,
                n_candidates=n_candidates,
                temperature=temperature,
                top_p=top_p,
            )
            return {
                output_gallery: new_images,
                generate_btn: gr.update(interactive=True),
            }

        upload_drag.upload(
            upload_image_fn,
            inputs=[upload_drag, image_list],
            outputs=[upload_drag, image_list, input_gallery, generate_btn],
        )
        clear_btn.click(
            clear_fn,
            inputs=None,
            outputs=[image_list, input_gallery, generate_btn, output_gallery],
        )
        generate_btn.click(
            disable_generate_btn,
            inputs=None,
            outputs=[generate_btn],
        ).then(
            generate_fn,
            inputs=[image_list, n_candidates_slider, gen_length_slider, temp_slider, top_p_slider],
            outputs=[output_gallery, generate_btn],
        )

        example_groups = load_example_image_groups('prompts')

        def add_image_group_fn(group_name, images):
            new_images = images + example_groups[group_name]
            return {
                image_list: new_images,
                input_gallery: new_images,
                generate_btn: gr.update(interactive=True),
            }

        gr.Markdown('## Default examples')
        for group_name, group_images in example_groups.items():
            with gr.Row():
                with gr.Column(scale=3):
                    add_button = gr.Button(value=f'Add {group_name}', elem_classes=['small-button'])
                with gr.Column(scale=7):
                    group_gallery = gr.Gallery(
                        value=[Image.fromarray(np.array(img)) for img in group_images],
                        columns=5,
                        rows=1,
                        object_fit='scale-down',
                        label=group_name,
                        elem_classes=['large-gallery'],
                    )

                add_button.click(
                    add_image_group_fn,
                    inputs=[gr.State(group_name), image_list],
                    outputs=[image_list, input_gallery, generate_btn],
                )

    demo.launch()

if __name__ == "__main__":
    mlxu.run(main)