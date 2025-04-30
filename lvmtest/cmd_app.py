import glob
import numpy as np
import mlxu
import os
import re
import torch
import logging
import os
import sys
from io import BytesIO
from natsort import natsorted
from PIL import Image
from datetime import datetime

from inference import LocalInferenceModel
from inference import MultiProcessInferenceModel

logger = logging.getLogger(__name__)

FLAGS, _ = mlxu.define_flags_with_default(
    dtype='float16',
    checkpoint='Emma02/LVM_ckpts',
    torch_devices='',
    context_frames=16,
    input_dir='images', # Add a flag for the input directory
    output_dir='output', # Add a flag for the output directory
)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def main(_):
    assert FLAGS.checkpoint != ''
    assert FLAGS.input_dir != ''  # Make sure input directory is provided

    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
    os.makedirs("logs", exist_ok=True)
    logfile = f"/mnt/home/av00440@ens.ad.etsmtl.ca/lvmtest/logs/logs_{timestamp}.log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(logfile, mode='a', encoding='utf-8')],
        level=logging.DEBUG
    )
    logger.info(f"Output directory set to: {logfile}")
    logger.info(f"Instanciating model")
    
    # Uncomment to use multi GPU
    # model = MultiProcessInferenceModel(
    #     checkpoint=FLAGS.checkpoint,
    #     torch_devices=[f'cuda:{i}' for i in range(torch.cuda.device_count())],
    #     dtype=FLAGS.dtype,
    #     context_frames=FLAGS.context_frames,
    #     use_lock=False,
    # )
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
        logger.info(f"Generating {n_new_frames} images with parameters \n"
                    f"Number of input images : {len(input_images)}\n"
                    f"Number of candidates : {n_candidates}\n"
                    f"Temperature :{temperature} \n"
                    f"Top p : {top_p} \n")
        assert len(input_images) > 0
        input_images = [
            np.array(img.convert('RGB').resize((256, 256)), dtype=np.float32) / 255.0
            for img in input_images
        ]
        input_images = np.stack(input_images, axis=0)
        output_images = model([input_images], n_new_frames, n_candidates, temperature, top_p)[0]

        logger.info(f"Output images : {len(output_images)}")
        
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

    def process_images_from_directory(input_dir, output_dir):
        # Find image files in the input directory
        image_files = glob.glob(os.path.join(input_dir, "*.png")) + \
                      glob.glob(os.path.join(input_dir, "*.jpg")) + \
                      glob.glob(os.path.join(input_dir, "*.jpeg"))
        image_files = natsorted(image_files, key=natural_sort_key)
        logger.info(f"Input images order :{image_files}")
        # Load images
        input_images = [Image.open(file) for file in image_files]

        # Generate images
        new_images = generate_images(input_images, n_new_frames=1, n_candidates=4, top_p=0.95)
        logger.info(f"New images {len(new_images)}")
        # Save generated images to the output directory
        os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
        for i, img in enumerate(new_images):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
            img.save(os.path.join(output_dir, f"generated_image_{i}_{timestamp}.png")) # Adjust filename format as needed

    # Call the processing function with flags
    process_images_from_directory(FLAGS.input_dir, FLAGS.output_dir)

if __name__ == "__main__":
    mlxu.run(main)
