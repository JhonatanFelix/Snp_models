import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas_plink import read_plink
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from LassoNet.constants import pheno_path, gen_path
import torch 
import pickle
import argparse
from datetime import datetime
from lassonet import LassoNetClassifier, plot_cv, plot_path
from lassonet.interfaces import LassoNetClassifierCV
from lassonet import LassoNetRegressor
import logging

def setup_logger(log_filename):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ],
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LassoNet model with configurable parameters"
    )

    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=20*(10**3),
        help="Number of samples to use from the dataset (default:20k)"

    )
    parser.add_argument(
        "-t",
        "--trait",
        type=int,
        required=True,
        help="Trait index to train on:" \
        " 2. " \
        " 3." \
        " 4." \
        " 5."
    )
    return parser.parse_args()


def main():
    # Loading arguments and logger
    args = parse_args()    
    log_filename = f"train_trait{args.trait}_n{args.n_samples}.log"

    setup_logger(log_filename)

    logging.info("Starting training")
    logging.info(f"Arguments: {vars(args)}")

    # Loading genomical and phenotypical data

    pheno_df = pd.read_csv(pheno_path, delimiter ='\t', header = None)

    pheno_df = pheno_df.drop(columns=[0])
    pheno_df = pheno_df.set_index(1)
    pheno_df.index.name == 'iid'

    bim, fam, bed = read_plink(gen_path)

    fam = fam.set_index('iid')

    # Separating data to training
    y_with_labels = pheno_df[args.trait][~pheno_df[args.trait].isna()]
    y = pheno_df[args.trait][~pheno_df[args.trait].isna()].to_numpy()

    ## Creating a mask to identify the numbers of rows relative to each identification number
    fam_pheno = fam['i'][~pheno_df[args.trait].isna()]

    X_complete = bed.T[fam_pheno] # TODO: I can make a function here to match the shape of the matrix with the output
    logging.info(f"The genotic data on the .bed matrix is on the shape:{X_complete.shape} \n")
    
    X = X_complete[:, :args.n_samples].compute()
    logging.info(f'We managed to compute the genetic matrix, now the new shape is: {X.shape}')
    del X_complete
    del fam_pheno
    del bim
    del fam
    del pheno_df

    ## Spliting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    del X
    
    logging.info("Training finished")


