import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from joblib import Parallel, delayed
from PIL import Image
from sklearn.linear_model import LogisticRegression
from vision_eval import mapper

import open_clip


def resize(imgs, size=(24, 24)):
    if type(imgs['img']) == Image.Image:
        imgs['img_resized'] = imgs['img'].resize(size)
        return imgs
    elif type(imgs['img']) == list:
        imgs['img_resized'] = [img.resize(size) for img in imgs['img']]
        return imgs

def encode_batch(model, dataloader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings.append(model.encode_image(batch))
    return torch.cat(embeddings, dim=0)

def preprocess_batch(dict, preprocessor, batch_size=64):
    prep = []
    for i in range(0, len(dict), batch_size):
        prep.append(preprocessor(dict[i:i+batch_size]['img_resized']).unsqueeze(0))
    
    return torch.cat(prep, dim=0)

def generate_embeddings(args):
    cifar = load_dataset("cifar10")
    if args.input_dir is None:
        cifar_train_resized = cifar['train'].map(resize, batched=True, num_proc=4, remove_columns=['img'])
        cifar_test_resized = cifar['test'].map(resize, batched=True, num_proc=4, remove_columns=['img'])

        cifar_train_resized.set_format(type='torch', columns=['img_resized'])
        cifar_test_resized.set_format(type='torch', columns=['img_resized'])

        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained=args.laion_model)
        prep_train = preprocess_batch(cifar_train_resized, preprocess)
        prep_test = preprocess_batch(cifar_test_resized, preprocess)

        # batch prep_train and prep_test and then run encode_image
        train_dataloader = DataLoader(prep_train, batch_size=64, shuffle=False)
        test_dataloader = DataLoader(prep_test, batch_size=64, shuffle=False)

        train_embeddings = encode_batch(model, train_dataloader)
        test_embeddings = encode_batch(model, test_dataloader)

        torch.save(train_embeddings, os.path.join(args.output_dir, "cifar_train_embeddings.pt"))
        torch.save(test_embeddings, os.path.join(args.output_dir, "cifar_test_embeddings.pt"))
    else:
        train_embeddings = torch.load(os.path.join(args.input_dir, "cifar_train_embeddings.pt"))
        test_embeddings = torch.load(os.path.join(args.input_dir, "cifar_test_embeddings.pt"))

        cifar_train_resized = cifar['train']
        cifar_test_resized = cifar['test']

    reg_thresholds = np.logspace(-6, 6, 96)

    max_score = 0.0

    results = Parallel(n_jobs=16, verbose=100)(
        delayed(mapper)(train_embeddings, cifar_train_resized['label'], test_embeddings, cifar_test_resized['label'], i) for i in reg_thresholds
    )

    print(results)
    max_tup = max(results, key=lambda x:x[1])
    print(f"Max tuple :: {max_tup}")
    

def main(args):
    generate_embeddings(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=False, help="Input directory for cached embeddings")
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--laion_model", type=str, required=False, default="laion400m_e32")
    args = parser.parse_args()
    main(args)
