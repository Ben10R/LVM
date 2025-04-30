import os
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
import numpy as np
from io import BytesIO
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Load the original dataset from Hugging Face Hub
print("Load dataset")
ds = load_dataset("rachit8562/mel_spectogram_bird_audio")

# Label to image paths mapping
label_to_image = {
    0: "dataset/Chlorischloris_1.jpg",
    1: "dataset/Columbapalumbus_1.jpg",
    2: "dataset/Corvusfrugilegus_1.jpg",
    3: "dataset/Delichonurbicum_1.jpg",
    4: "dataset/Dendrocoposmajor_1.jpg",
    5: "dataset/Passermontanus_1.jpg",
    6: "dataset/Phoenicurusochruros_1.jpg",
    7: "dataset/Sittaeuropaea_1.jpg",
    8: "dataset/Turdusmerula_1.jpg",
    9: "dataset/Turduspilaris_1.jpg"
}

# Function to load image and convert to a NumPy array
def image_to_numpy(image_path):
    try:
        # Open the image using PIL and convert to RGB
        img = Image.open(image_path).convert('RGB').resize((600, 600))
        # Convert the image to a NumPy array
        img_array = np.array(img)
        return img_array
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {e}")
        return None

# Replace numeric labels with image data in memory
def process_data(dataset):
    input_images = []
    output_images = []

    for i in range(len(dataset)):
        try:
            # Get the image path from label_to_image
            image_path = label_to_image.get(dataset[i]["label"])
            if image_path:
                # Load the actual image and convert it to a NumPy array
                img_array = image_to_numpy(image_path)
                if img_array is not None:
                    input_images.append(dataset[i]["image"])  # Keep original input image
                    output_images.append(img_array)  # Store the image array as output
            else:
                continue
            if(i%1000 == 0):
                print(f"Step {i}")
        except Exception as e:
            logger.error(f"Error at index {i}: {e}")
            continue

    # Create a new dataset with processed data
    return Dataset.from_dict({
        "image": input_images,
        "label": output_images
    })

# Process the train and test datasets
train_dataset = process_data(ds["train"])
train_dataset.save_to_disk("lvm-bird-train", writer_batch_size=50)
print("Processed train dataset")
test_dataset = process_data(ds["test"])
test_dataset.save_to_disk("lvm-bird-test", writer_batch_size=50)
print("Processed test dataset")
# Combine into a DatasetDict if needed (train/test splits)
final_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})
print("Final dataset merged, writing to disk")
final_dataset.save_to_disk("lvm_bird_spectrogram", writer_batch_size=50)
print("Final dataset merged, pushing to hub")
# Upload to Hugging Face Hub
final_dataset.push_to_hub("Ben10x/lvm_bird_spectrogram",writer_batch_size=50)

print("Dataset processing complete and pushed to Hugging Face Hub.")