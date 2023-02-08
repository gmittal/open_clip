import argparse
import os

import numpy as np
import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from joblib import Parallel, delayed


def mapper(train_tensors, train_targets, val_tensors, val_targets, reg_value):
    clf = LogisticRegression(random_state=0, C=reg_value, max_iter=1000, verbose=0).fit(
            train_tensors, train_targets
        )

    y_pred = clf.score(val_tensors, val_targets)


    return (reg_value, y_pred)


def main(args):

    train_tensors = []
    train_targets = []
    train_files = os.listdir(os.path.join(args.input_dir, "vissl_image_train"))
    for f in tqdm.tqdm(train_files):
        train_tensors.append(
            torch.load(os.path.join(args.input_dir, "vissl_image_train", f))
        )
        train_targets.append(int(f.split(".")[0].split("_")[1]))

    print("Loaded train set", len(train_tensors))

    val_tensors = []
    val_targets = []
    val_files = os.listdir(os.path.join(args.input_dir, "vissl_image_test"))
    for f in tqdm.tqdm(val_files):
        val_tensors.append(
            torch.load(os.path.join(args.input_dir, "vissl_image_test", f))
        )
        val_targets.append(int(f.split(".")[0].split("_")[1]))

    print("Loaded test set", len(val_tensors))

    results = []

    reg_thresholds = np.logspace(-6, 6, 96)

    max_score = 0.0

    results = Parallel(n_jobs=16, verbose=100)(
        delayed(mapper)(train_tensors, train_targets, val_tensors, val_targets, i) for i in reg_thresholds
    )

    print(results)
    max_tup = max(results, key=lambda x:x[1])
    print(f"Max tuple :: {max_tup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="save")
    args = parser.parse_args()
    main(args)