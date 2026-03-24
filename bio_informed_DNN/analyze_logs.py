import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = "./logs_models/ARS-UCD2.0.115/critHuberLoss_gelu/"
OUTPUT_DIR = "./log_analysis_results/ARS-UCD2.0.115/critHuberLoss_gelu/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------
# extract hyperparameters
# -----------------------------------


def parse_filename(filename):
    basename = os.path.basename(filename)

    pattern = (
        r"training_\[(?P<layers>.*?)\]"
        r"_lr(?P<lr>[\d\.e-]+)"
        r"_trait(?P<trait>\d+)"
        r"_epoch(?P<epoch>\d+)"
        r"_crit(?P<crit>.*?)"
        r"_act(?P<act>.*?)"
        r"_batch(?P<batch>\d+)"
        r"_wdecay(?P<wdecay>[\d\.e-]+)"
        r"_dropout(?P<dropout>[\d\.]+)"
        r"_seed(?P<seed>\d+)"
        r"_ea(?P<ea>True|False)"
    )

    match = re.search(pattern, basename)
    if not match:
        return None

    params = match.groupdict()

    # Remove crit and act
    params.pop("crit", None)
    params.pop("act", None)

    return params


# -----------------------------------
# extract final MAE and Correlation
# -----------------------------------


def extract_final_metrics(filepath):
    mae = None
    corr = None

    with open(filepath, "r") as f:
        for line in f:
            if "MAE:" in line:
                mae = float(line.strip().split("MAE:")[-1])
            if "Pearson Correlation:" in line:
                corr = float(line.strip().split("Pearson Correlation:")[-1])

    return mae, corr


# -----------------------------------
# Load all logs
# -----------------------------------

records = []

for filepath in glob.glob(os.path.join(LOG_DIR, "*.log")):
    params = parse_filename(filepath)
    if params is None:
        continue

    mae, corr = extract_final_metrics(filepath)

    if mae is None or corr is None:
        continue

    params["MAE"] = mae
    params["Correlation"] = corr
    records.append(params)

df = pd.DataFrame(records)

if df.empty:
    print("No valid logs found.")
    exit()

# Convert numeric columns where possible
for col in df.columns:
    if col not in ["ea", "layers"]:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

print("\nTotal experiments found:", len(df))


# -----------------------------------
# best results
# -----------------------------------

best_mae_row = df.loc[df["MAE"].idxmin()]
best_corr_row = df.loc[df["Correlation"].idxmax()]

print("\n===== BEST MAE =====")
print(best_mae_row)

print("\n===== BEST CORRELATION =====")
print(best_corr_row)


# -----------------------------------
# aalyze hyperparameters
# -----------------------------------

HYPERPARAMS = [col for col in df.columns if col not in ["MAE", "Correlation"]]


def create_bar_plot(stats, param, metric_mean, metric_std, ylabel):

    # Sort numeric parameters
    try:
        stats = stats.sort_values(by=param)
    except:
        pass

    x_labels = stats[param].astype(str)
    means = stats[metric_mean]
    stds = stats[metric_std]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(x_labels, means, yerr=stds, capsize=5)

    plt.xlabel(param)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {param}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{param}_{ylabel.replace(' ', '_')}.png"))
    plt.close()


for param in HYPERPARAMS:
    print(f"\n\n===== Analyzing {param} =====")

    grouped = df.groupby(param)

    stats = grouped.agg(
        mean_MAE=("MAE", "mean"),
        std_MAE=("MAE", "std"),
        var_MAE=("MAE", "var"),
        mean_Corr=("Correlation", "mean"),
        std_Corr=("Correlation", "std"),
        var_Corr=("Correlation", "var"),
    ).reset_index()

    print(stats)

    # Save stats table
    stats.to_csv(os.path.join(OUTPUT_DIR, f"{param}_statistics.csv"), index=False)

    # Publication-style bar plots
    create_bar_plot(stats, param, "mean_MAE", "std_MAE", "Mean MAE")
    create_bar_plot(stats, param, "mean_Corr", "std_Corr", "Mean Correlation")

print("\nAnalysis complete.")
print("Results saved in:", OUTPUT_DIR)
