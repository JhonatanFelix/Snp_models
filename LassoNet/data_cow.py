import pandas as pd
import numpy as np
from pandas_plink import read_plink

from sklearn.preprocessing import StandardScaler


def load_cow_data(pheno_path_nan, pheno_path, gen_path, trait=2):
    # Loading genomical and phenotypical data

    def get_pheno_df(pheno_path):
        pheno_df = pd.read_csv(pheno_path, delimiter="\t", header=None)
        
        pheno_df = pheno_df.drop(columns=[0])
        pheno_df = pheno_df.set_index(1)
        pheno_df.index.name == "iid"
        return pheno_df

    pheno_df_nan = get_pheno_df(pheno_path_nan)
    pheno_df = get_pheno_df(pheno_path)

    bim, fam, bed = read_plink(gen_path)

    fam = fam.set_index("iid")

    # Separating data to training
    y_train = pheno_df[trait][~pheno_df_nan[trait].isna()].to_numpy()
    y_test = pheno_df[trait][pheno_df_nan[trait].isna()].to_numpy()
    y_test = y_test[~np.isnan(y_test)]

    pheno_df["i"] = fam["i"]

    ## Creating a mask to identify the numb~ers of rows relative to each identification number
    fam_pheno = fam["i"][~pheno_df_nan[trait].isna()]
    fam_pheno_test = pheno_df[pheno_df_nan[trait].isna()]
    fam_pheno_test = fam_pheno_test["i"][~fam_pheno_test[trait].isna()]


    X_train = bed.T[
        fam_pheno
    ]

    X_test = bed.T[
        fam_pheno_test
    ]

    del pheno_df_nan
    del fam_pheno
    del fam_pheno_test
    del bim
    del fam
    del pheno_df
    
    scaler_X  = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1)

    ## Spliting the dataset
    return X_train, y_train, X_test, y_test
