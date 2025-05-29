import json
import os

import torch
from tqdm import tqdm

from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from utils import torch_load, train_diag_fim_logtr

samples_nr = 500  # How many per-example gradients to accumulate


def eval(dataset_name, loader, model):
    # Initialize variables for evaluation
    correct = 0
    total = 0

    # Start evaluation loop
    print()
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Evaluating {dataset_name}")
        for images, labels in progress_bar:
            images, labels = images.to(args.device), labels.to(args.device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Update the count of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Optionally update the progress bar with accuracy
            accuracy = 100 * correct / total
            progress_bar.set_postfix({"accuracy": accuracy})

    # Final accuracy
    accuracy = 100 * correct / total
    logdet_hF = train_diag_fim_logtr(args, model, dataset_name, samples_nr)

    return accuracy, logdet_hF


def eval_single_task(args, dataset_name, model_path):
    results_path = os.path.join(
        args.save,
        f"{dataset_name}_{'finetuned' if model_path else 'base'}_results.json",
    )
    if os.path.exists(results_path):
        print(f"Results already exist at {results_path}. Skipping evaluation.")
        return
    encoder = ImageEncoder(args=args).to(args.device)

    if model_path is not None:
        print(f"Loading model from {model_path}")
        state_dict = torch_load(model_path, device=args.device).state_dict()
        encoder.load_state_dict(state_dict)

    head = get_classification_head(args, dataset_name + "Val")
    model = ImageClassifier(encoder, head).to(args.device)
    model.eval()  # Set the model to evaluation mode

    # Validation
    dataset = get_dataset(
        dataset_name + "Val",
        preprocess=model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    loader = get_dataloader(
        dataset,
        is_train=False,
        args=args,
    )
    accuracy, logdet_hF = eval(dataset_name, loader, model)
    print(f"Validation Dataset: {accuracy:.2f} - logdet_hF: {logdet_hF:.3f}")
    # Save validation results to JSON
    val_results = {
        "dataset": "Validation",
        "accuracy": accuracy,
        "logdet_hF": logdet_hF,
    }

    # Test
    dataset = get_dataset(
        dataset_name,
        preprocess=model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    loader = get_dataloader(
        dataset,
        is_train=False,
        args=args,
    )
    accuracy, logdet_hF = eval(dataset_name, loader, model)
    print(f"Test Dataset: {accuracy:.2f} - logdet_hF: {logdet_hF:.3f}")
    test_results = {
        "dataset": "Test",
        "accuracy": accuracy,
        "logdet_hF": logdet_hF,
    }

    with open(results_path, "w") as f:
        json.dump(test_results, f)


if __name__ == "__main__":
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    args = parse_arguments()
    if os.path.exists(os.path.join(args.save, "results.json")):
        print("Results already exist. Skipping evaluation.")
        exit(0)
    print()
    print("Evaluating pretrained model")
    for dataset in datasets:
        eval_single_task(args=args, dataset_name=dataset, model_path=None)
    print()
    print("Evaluating fine-tuned models")
    for dataset in datasets:
        eval_single_task(
            args=args,
            dataset_name=dataset,
            model_path=os.path.join(args.save, f"finetuned_{dataset}.pt"),
        )
