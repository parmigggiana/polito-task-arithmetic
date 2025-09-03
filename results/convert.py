import json
import pandas as pd

# Mapping of filenames to phase names
file_map = {
    "pre_trained_results_wd1.json": "Pre-train",
    "before_scaling_results_wd1.json": "Finetuned",
    "scaled_results_wd1.json": "Scaled",
    "addition_results_wd1.json": "Merged"
}

# Metric key mapping
metric_alias = {
    "accuracy": "Accuracy",
    "abs_accuracy": "Abs Accuracy",
    "norm_accuracy": "Norm Accuracy",
    "logdet_hF": "logdet_hF"
}

# Dataset names
datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

# List to collect rows
rows = []

# Iterate over each file and phase
for file_name, phase in file_map.items():
    with open(file_name, "r") as f:
        data = json.load(f)

    for split in ["train", "test"]:
        for key in metric_alias:
            row = {
                "Split": split.capitalize(),
                "Metric": metric_alias[key],
                "Phase": phase,
                "wd": "0.1"
            }
            present = False
            for dataset in datasets:
                try:
                    value = data[dataset][split][key]
                    row[dataset] = f"{value:.2f}"
                    present = True
                except KeyError:
                    row[dataset] = ""
            if present:
                rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Ensure consistent column order
columns = ["Split", "Metric", "Phase", "wd"] + datasets
df = df[columns]

# Save as TSV for Google Sheets
df.to_csv("formatted_results.tsv", index=False, sep='\t')
print("Saved to formatted_results.tsv")

