import json
import os

import numpy as np

from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from finetune import eval, eval_acc
from heads import get_classification_head
from modeling import ImageClassifier
from task_vectors import NonLinearTaskVector

datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]


def average_normalized_accuracy(args, task_vectors, pretrained_model_path, alpha):
    acc = 0.0
    encoder_merged = sum(task_vectors).apply_to(
        pretrained_model_path, scaling_coef=alpha
    )

    for i in range(len(task_vectors)):
        task_vector = task_vectors[i]
        encoder_single_task = task_vector.apply_to(pretrained_model_path)

        classification_head = get_classification_head(args, datasets[i] + "Val")
        model_single_task = ImageClassifier(
            encoder_single_task, classification_head
        ).to(args.device)
        model_merged = ImageClassifier(encoder_merged, classification_head).to(
            args.device
        )

        model_single_task.eval()
        model_merged.eval()

        dataset = get_dataset(
            datasets[i] + "Val",
            preprocess=model_single_task.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )

        loader = get_dataloader(
            dataset,
            is_train=False,
            args=args,
        )
        accuracy_s = eval_acc(args, loader, model_single_task)

        loader = get_dataloader(
            dataset,
            is_train=False,
            args=args,
        )
        accuracy_m = eval_acc(args, loader, model_merged)
        print()
        print(
            f"Dataset: {datasets[i]}, Single Task Accuracy: {accuracy_s:.4f}, Merged Accuracy: {accuracy_m:.4f}"
        )
        acc += accuracy_m / accuracy_s

    return acc / len(task_vectors)


def find_alpha(args, task_vectors, pretrained_model_path):
    accs = []
    print("Finding optimal alpha...")
    for alpha_candidate in np.arange(0.0, 1.01, 0.05):
        acc = average_normalized_accuracy(
            args, task_vectors, pretrained_model_path, alpha_candidate
        )
        print(f"Alpha: {alpha_candidate:.2f}, Average Normalized Accuracy: {acc:.4f}")
        accs.append(acc)

    return accs.index(max(accs)) * 0.05  # Return the alpha that gives the best accuracy


def eval_task_addition(args):
    """
    Evaluates a model with applied task vectors across multiple datasets.
    """
    addition_path = os.path.join(args.save, "addition_results.json")
    scaled_path = os.path.join(args.save, "scaled_results.json")

    if os.path.exists(addition_path) and os.path.exists(scaled_path):
        print("Results already exist. Skipping evaluation.")
        return

    # print("Selected Device: " + args.device)

    # print(
    #     f"Evaluating task addition model with data from {data_location} and saving to {save}"
    # )

    print()
    print()
    task_vectors = []

    for dataset_name in datasets:
        # print(f"Creating task vector for {dataset_name}...")
        pretrained_model_path = os.path.join(args.save, "base.pt")
        finetuned_model_path = os.path.join(args.save, f"finetuned_{dataset_name}.pt")
        task_vector = NonLinearTaskVector(
            pretrained_checkpoint=pretrained_model_path,
            finetuned_checkpoint=finetuned_model_path,
        )

        task_vectors.append(task_vector)  # Store the task vector

    if args.alpha is None:
        alpha = find_alpha(
            args, task_vectors=task_vectors, pretrained_model_path=pretrained_model_path
        )
        print(f"Optimal alpha found: {alpha}")
    else:
        alpha = args.alpha
        print(f"Using provided alpha: {alpha}")

    merged_encoder = sum(task_vectors).apply_to(
        pretrained_model_path, scaling_coef=alpha
    )

    metrics_after_addition = {}
    metrics_after_scaling = {}

    # Load finetuned accuracy from eval_single_task results
    single_task_results_path = os.path.join(args.save, "before_scaling_results.json")
    with open(single_task_results_path, "r") as f:
        single_task_results = json.load(f)

    for dataset_name in datasets:
        scaled_encoder = (
            task_vectors[datasets.index(dataset_name)]
            .apply_to(pretrained_model_path, scaling_coef=alpha)
            .to(args.device)
        )

        classification_head = get_classification_head(args, dataset_name + "Val")
        scaled_model = ImageClassifier(scaled_encoder, classification_head).to(
            args.device
        )
        merged_model = ImageClassifier(merged_encoder, classification_head).to(
            args.device
        )

        scaled_model.eval()
        merged_model.eval()

        dataset = get_dataset(
            dataset_name,
            preprocess=merged_model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        loader = get_dataloader(
            dataset,
            is_train=False,
            args=args,
        )
        acc_merged = eval_acc(args, loader, merged_model)

        loader = get_dataloader(
            dataset,
            is_train=False,
            args=args,
        )
        acc_scaled = eval_acc(args, loader, scaled_model)

        acc_finetuned = single_task_results[dataset_name]["test"]["accuracy"]

        norm_accuracy = 100 * (acc_merged / acc_finetuned)

        metrics_after_addition[dataset_name] = {
            "train": {},
            "test": {"abs_accuracy": acc_merged, "norm_accuracy": norm_accuracy},
        }

        metrics_after_scaling[dataset_name] = {
            "train": {},
            "test": {
                "accuracy": acc_scaled,
            },
        }
        print(f"Dataset: {dataset_name}, Merged Accuracy: {acc_merged:.4f}")

        dataset = get_dataset(
            dataset_name + "Val",
            preprocess=merged_model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        loader = get_dataloader(
            dataset,
            is_train=True,
            args=args,
        )
        acc_merged, logdet_merged = eval(args, loader, dataset_name, merged_model)

        loader = get_dataloader(
            dataset,
            is_train=True,
            args=args,
        )
        acc_scaled, logdet_scaled = eval(args, loader, dataset_name, scaled_model)

        acc_finetuned = single_task_results[dataset_name]["train"]["accuracy"]
        norm_accuracy = 100 * (acc_merged / acc_finetuned)

        metrics_after_addition[dataset_name]["train"] = {
            "abs_accuracy": acc_merged,
            "norm_accuracy": norm_accuracy,
            "logdet_hF": logdet_merged,
        }

        metrics_after_scaling[dataset_name]["train"] = {
            "accuracy": acc_scaled,
            "logdet_hF": logdet_scaled,
        }

    metrics_after_addition["alpha"] = alpha
    metrics_after_scaling["alpha"] = alpha

    with open(addition_path, "w") as f:
        json.dump(metrics_after_addition, f, indent=4)

    with open(scaled_path, "w") as f:
        json.dump(metrics_after_scaling, f, indent=4)


def main():
    args = parse_arguments()
    eval_task_addition(args)


if __name__ == "__main__":
    main()
