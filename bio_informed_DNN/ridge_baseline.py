import argparse
import json
import os

import numpy as np
import pandas as pd
from pandas_plink import read_plink
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===============================
# Arguments
# ===============================


def parse_args():
    parser = argparse.ArgumentParser(description="Ridge Regression baseline")

    parser.add_argument(
        "-t", "--trait", type=int, default=2, help="Trait index to train on"
    )

    return parser.parse_args()


# ===============================
# Main
# ===============================


def main():

    args = parse_args()

    gen_path = "../data/ML/BBB2023_MD"
    bim, _, bed = read_plink(gen_path)

    X = bed

    # ===============================
    # Load phenotype
    # ===============================

    pheno_df = pd.read_csv(
        "../data/ML/pheno_2023bbb_0twins_6traits_mask", delimiter="\t", header=None
    )

    pheno_df = pheno_df.drop(columns=[0]).set_index(1)

    y = pheno_df[args.trait][~pheno_df[args.trait].isna()].to_numpy()
    X = X.T[~pheno_df[args.trait].isna()].compute()

    print("Data loaded.")

    # ===============================
    # Split (same as NN)
    # ===============================

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1000, random_state=42
    )

    # ===============================
    # Scaling
    # ===============================

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    print("Scaling complete.")

    # ===============================
    # Ridge CV (Automatic alpha search)
    # ===============================

    print("Searching best alpha...")

    # Log-scale search space
    alphas = np.logspace(-4, 4, 50)

    ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error", cv=5)

    ridge.fit(X_train, y_train)

    best_alpha = ridge.alpha_

    print(f"Best alpha found: {best_alpha}")

    # ===============================
    # Predictions
    # ===============================

    y_pred_scaled = ridge.predict(X_val)

    # Inverse scaling
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_val_real = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1))

    # ===============================
    # Metrics
    # ===============================

    mae = mean_absolute_error(y_val_real, y_pred_real)
    mse = mean_squared_error(y_val_real, y_pred_real)
    r2 = r2_score(y_val_real, y_pred_real)
    pearson_corr, _ = pearsonr(y_val_real.flatten(), y_pred_real.flatten())

    print("\n===== FINAL RIDGE RESULTS =====")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")
    print(f"Pearson Correlation: {pearson_corr}")

    # ===============================
    # Save JSON
    # ===============================

    results = {
        "trait": args.trait,
        "best_alpha": float(best_alpha),
        "MAE": float(mae),
        "MSE": float(mse),
        "R2": float(r2),
        "Pearson_correlation": float(pearson_corr),
    }

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"ridge_results_trait{args.trait}.json",
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
