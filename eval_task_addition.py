import argparse
import os

from heads import get_classification_head
from modeling import ImageClassifier
import torch
from torchvision import transforms
from tqdm import tqdm

from datasets.common import get_dataloader
from datasets.registry import get_dataset
from task_vectors import NonLinearTaskVector

from args import parse_arguments
from eval_single_task import eval

import numpy as np

datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

def average_normalized_accuracy(args, task_vectors, pretrained_model_path, alpha):
    acc = 0.0
    for i in range(len(task_vectors)):
        task_vector = task_vectors[i]
        encoder_single_task = task_vector.apply_to(pretrained_model_path, scaling_coef=alpha)
        encoder_cumulative = sum(task_vectors[:i + 1]).apply_to(pretrained_model_path, scaling_coef=alpha)

        classification_head = get_classification_head(args, datasets[i] + "Val")
        model_single_task = ImageClassifier(encoder_single_task, classification_head).to(args.device)
        model_cumulative = ImageClassifier(encoder_cumulative, classification_head).to(args.device)

        model_single_task.eval()
        model_cumulative.eval()

        acc = 0.0
        dataset_name = datasets[i]
        dataset = get_dataset(
            dataset_name + "Val",
            preprocess=model_single_task.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        loader = get_dataloader(
            dataset,
            is_train=False,
            args=args,
        )

        accuracy_c = eval(args, dataset_name, loader, model_cumulative)
        accuracy_m = eval(args, dataset_name, loader, model_single_task)

        acc += accuracy_c / accuracy_m

    return acc / (i + 1)



def find_alpha(args, task_vectors, pretrained_model_path):
    accs = []
    print("Finding optimal alpha...")
    for alpha_candidate in np.arange(0.0, 1.01, 0.05):
        acc = average_normalized_accuracy(args, task_vectors, pretrained_model_path,alpha_candidate)
        print(f"Alpha: {alpha_candidate:.2f}, Average Normalized Accuracy: {acc:.4f}")
        accs.append(acc)


    return accs.index(max(accs)) * 0.05  # Return the alpha that gives the best accuracy

def eval_task_addition(args):
    """
    Evaluates a model with applied task vectors across multiple datasets.
    """

    data_location=args.data_location
    save=args.save

    print(
        f"Evaluating task addition model with data from {data_location} and saving to {save}"
    )

    task_vectors = []

    for dataset_name in datasets:
        print(f"Creating task vector for {dataset_name}...")
        pretrained_model_path=os.path.join(args.save, f'base.pt')
        finetuned_model_path = f"./out/finetuned_{dataset_name}.pt"
        task_vector = NonLinearTaskVector(pretrained_checkpoint=pretrained_model_path,
                                          finetuned_checkpoint=finetuned_model_path)

        task_vectors.append(task_vector)  # Store the task vector

    alpha = find_alpha(args, task_vectors=task_vectors, pretrained_model_path=pretrained_model_path)
    print(f"Optimal alpha found: {alpha}")

    merged_model = sum(task_vectors).apply_to(pretrained_model_path, scaling_coef=alpha)

    results = []
    for dataset_name in datasets:
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
        abs_accuracy, _ = eval(args, dataset_name, loader, merged_model)

        finetuned_encoder = task_vectors[datasets.index(dataset_name)].apply_to(pretrained_model_path, scaling_coef=alpha)
        classification_head = get_classification_head(args, dataset_name + "Val")
        finetuned_model = ImageClassifier(finetuned_encoder, classification_head).to(args.device)
        abs_accuracy_finetuned, _ = eval(dataset_name, loader, finetuned_model)

        norm_accuracy = abs_accuracy / abs_accuracy_finetuned if abs_accuracy_finetuned != 0 else 0.0

        results.append({
            "dataset": dataset_name,
            "abs_accuracy": abs_accuracy,
            "norm_accuracy": norm_accuracy,
        })

    avg_absolute_acc = sum(result["abs_accuracy"] for result in results) / len(results)
    avg_normalized_acc = sum(result["norm_accuracy"] for result in results) / len(results)


    print(f"Average Absolute Accuracy: {avg_absolute_acc:.2f}")
    print(f"Average Normalized Accuracy: {avg_normalized_acc:.2f}")


def main():
    args = parse_arguments()
    eval_task_addition(args)


if __name__ == "__main__":
    main()
