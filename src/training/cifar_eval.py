import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from joblib import Parallel, delayed
from PIL import Image
from sklearn.linear_model import LogisticRegression
from vision_eval import mapper

import open_clip


def resize(imgs, size=(224, 224)):
    imgs['img_resized'] = imgs['img'].resize(size)
    return imgs

def generate_embeddings(args):
    cifar = load_dataset("cifar10")
    if args.input_dir is None:
        cifar_train_resized = cifar['train'].map(resize, batched=True, num_proc=4, remove_columns=['img'])
        cifar_test_resized = cifar['test'].map(resize, batched=True, num_proc=4, remove_columns=['img'])

        cifar_train_resized.set_format(type='torch', columns=['img_resized'])
        cifar_test_resized.set_format(type='torch', columns=['img_resized'])

        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained=args.laion_model)
        prep_train = preprocess(cifar_train_resized['img_resized'])
        prep_test = preprocess(cifar_test_resized['img_resized'])

        train_embeddings = model.encode_image(prep_train)
        test_embeddings = model.encode_image(prep_test)

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
    parser.add_argument("--input_dir", type=str, required=False, description="Input directory for cached embeddings")
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--laion_model", type=str, required=False, default="laion400m_e32")
    args = parser.parse_args()
    main(args)
