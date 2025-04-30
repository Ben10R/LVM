
import os
from functools import partial
from tempfile import NamedTemporaryFile
import random
import json
from base64 import b64encode
from tqdm import tqdm, trange

import numpy as np
import mlxu
from datasets import load_dataset
import torch

from vqvae_muse import VQGANModel, get_tokenizer_muse
from utils import read_image_to_tensor

from PIL import Image
from io import BytesIO

import logging 
logger = logging.getLogger(__name__)

FLAGS, _ = mlxu.define_flags_with_default(
    input_image_dir='',
    output_image_dir='',
    output_file='./dataset.jsonl',
    input_filter_key='',
    output_filter_key='',
    input_suffix='',
    output_suffix='',
    crop='',
    batch_size=1,
    n_shots=8,
    n_epochs=5,
    n_workers=2,
    dtype='fp32',
)


class PairedImageDataset(torch.utils.data.Dataset):

    def __init__(self, input_images, output_images):

        self.input_images = input_images
        self.output_images = output_images

    def __getitem__(self, index):
        try:
            return (
                read_image_to_tensor(self.input_images[index]),
                read_image_to_tensor(self.output_images[index])
            )
        except Exception as e:
            print(f'Error: {e} for {self.input_images[index]}')
            return self[np.random.randint(0, len(self))]

    def __len__(self):
        return len(self.input_images)


def main(argv):
    # assert FLAGS.input_image_dir != ''
    # assert FLAGS.output_image_dir != ''
    assert FLAGS.output_file != ''

    # Load the pre-trained vq model from the hub
    logger.info("Load tokenizer")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = get_tokenizer_muse()
    net.to(device)
    # net = VQGANModel.from_pretrained('vqlm/muse/ckpts/laion').to(device)
    # net.eval()

    # Load the dataset from hub
    logger.info("Load dataset")
    ds = load_dataset("rachit8562/mel_spectogram_bird_audio")
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
    input_images = []
    output_images = []
        
    for i in range(len(ds["train"])):
        print(i)
        try:
            input_images.append(ds["train"][i]["image"]) 
            output_images.append(label_to_image[ds["train"][i]["label"]])
        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue
    
    # input_images = os.listdir(FLAGS.input_image_dir)
    # input_images = [i for i in input_images if i.endswith('.png') or i.endswith('.jpg') or i.endswith('.jpeg')]
    # input_images = [i for i in input_images if FLAGS.input_filter_key in i]
    # input_images = sorted(input_images)
    # output_images = os.listdir(FLAGS.output_image_dir)
    # output_images = [i for i in output_images if i.endswith('.png') or i.endswith('.jpg') or i.endswith('.jpeg')]
    # output_images = [i for i in output_images if FLAGS.output_filter_key in i]
    # output_images = sorted(output_images)
    logger.info(f"Len input :{len(input_images)} | Len output {len(output_images)}")
    assert len(input_images) == len(output_images)


    # input_images = [
    #     os.path.join(FLAGS.input_image_dir, s)
    #     for s in input_images
    # ]
    # output_images = [
    #     os.path.join(FLAGS.output_image_dir, s)
    #     for s in output_images
    # ]

    dataset = PairedImageDataset(input_images, output_images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size * FLAGS.n_shots,
        shuffle=False,
        num_workers=FLAGS.n_workers,
        drop_last=True
    )

    total_images = len(input_images) - len(input_images) % (FLAGS.batch_size * FLAGS.n_shots)
    logger.info(f"Total images : {total_images}")
    with torch.no_grad():
        with NamedTemporaryFile() as ntf:
            all_tokens = np.memmap(ntf, dtype='i4', mode='w+', shape=(total_images, 512))
            all_tokens[:] = 0

            index = 0
            for input_image_batch, output_image_batch in tqdm(dataloader, ncols=0):
                _, input_token_batch = net.encode(input_image_batch.permute(0,3,1,2).to(device))
                _, output_token_batch = net.encode(output_image_batch.permute(0, 3, 1, 2).to(device))


                all_tokens[index:index + input_image_batch.shape[0]] = np.concatenate(
                    [input_token_batch.cpu().numpy().astype(np.int32), output_token_batch.cpu().numpy().astype(np.int32)],
                    axis=1
                )
                index += input_image_batch.shape[0]

            with open(FLAGS.output_file, 'w') as fout:
                logger.info(f"Printing to file : {FLAGS.output_file}")
                for _ in trange(FLAGS.n_epochs, ncols=0):
                    indices = np.random.permutation(total_images).reshape(-1, FLAGS.n_shots)
                    for i in trange(indices.shape[0], ncols=0):
                        tokens = all_tokens[indices[i], :].reshape(-1)
                        data = {'tokens': b64encode(tokens.tobytes()).decode('utf-8'),}
                        fout.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    mlxu.run(main)