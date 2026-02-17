import argparse
import logging
import pickle

import numpy as np
import pandas as pd
import torch
from constants import gen_path, pheno_path
from lassonet import LassoNetRegressor
from pandas_plink import read_plink
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def setup_logger(log_filename):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LassoNet model with configurable parameters"
    )

    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=20 * (10**3),
        help="Number of samples to use from the dataset (default:20k)",
    )
    parser.add_argument(
        "-t",
        "--trait",
        type=int,
        required=True,
        help="Trait index to train on: 2.  3. 4. 5.",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=int,
        nargs="+",  # this means one or more integers
        required=False,
        default=[10],
        help="Hidden layer sizes (e.g. 10 20) (default:10)",
    )
    return parser.parse_args()


def main():
    # Loading arguments and logger
    args = parse_args()
    layers = tuple(args.layers)

    log_filename = f"train_trait{args.trait}_n{args.n_samples}_layers{layers}.log"

    setup_logger(log_filename)

    logging.info("Starting training")
    logging.info(f"Arguments: {vars(args)}")

    # Loading genomical and phenotypical data

    pheno_df = pd.read_csv(pheno_path, delimiter="\t", header=None)

    pheno_df = pheno_df.drop(columns=[0])
    pheno_df = pheno_df.set_index(1)
    pheno_df.index.name == "iid"

    bim, fam, bed = read_plink(gen_path)

    fam = fam.set_index("iid")

    # Separating data to training
    y = pheno_df[args.trait][~pheno_df[args.trait].isna()].to_numpy()

    ## Creating a mask to identify the numbers of rows relative to each identification number
    fam_pheno = fam["i"][~pheno_df[args.trait].isna()]

    X_complete = bed.T[
        fam_pheno
    ]  # TODO: I can make a function here to match the shape of the matrix with the output
    logging.info(
        f"The genotic data on the .bed matrix is on the shape:{X_complete.shape} \n"
    )

    X = X_complete[:, : args.n_samples].compute()
    logging.info(
        f"We managed to compute the genetic matrix, now the new shape is: {X.shape}"
    )
    del X_complete
    del fam_pheno
    del bim
    del fam
    del pheno_df

    ## Spliting the dataset
    logging.info("Splitting the dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    del X

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    optim_fn = lambda params: torch.optim.Adam(params, lr=1e-4)

    model = LassoNetRegressor(
        hidden_dims=layers,
        verbose=True,
        patience=(100, 5),
        optim=(optim_fn, optim_fn),
        lambda_start=5,
    )

    logging.info("Calculating the path of the model")
    path = model.path(X_train, y_train, return_state_dicts=True)

    data_to_save = {
        "path": path,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    del X_train
    del path
    del X_test
    del y_train
    del y_test

    logging.info("Saving path of the model")
    with open(
        f"lassonet_{int(args.n_samples / 1000)}k_n{args.trait}_layers{layers}.pkl", "wb"
    ) as f:
        pickle.dump(data_to_save, f)

    logging.info("Training finished")


if __name__ == "__main__":
    main()
