import json
import os

from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from finetune import eval, eval_acc
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from utils import torch_load, train_diag_fim_logtr


def eval_single_task(args, dataset_name, model_path):
    print()
    encoder = ImageEncoder(args=args).to(args.device)
    # print(f"Loading model from {model_path}")
    state_dict = torch_load(model_path, device=args.device).state_dict()
    encoder.load_state_dict(state_dict)

    head = get_classification_head(args, dataset_name + "Val")
    model = ImageClassifier(encoder, head).to(args.device)
    model.eval()  # Set the model to evaluation mode

    # Metrics before scaling & addition
    dataset = get_dataset(
        dataset_name + "Val",
        preprocess=model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    loader = get_dataloader(
        dataset,
        is_train=True,
        args=args,
    )
    accuracy, logdet_hF = eval(args, loader, dataset_name, model)
    print(f"Training Dataset | accuracy: {accuracy:.2f} - logdet_hF: {logdet_hF:.3f}")
    train_results = {
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
    accuracy = eval_acc(args, loader, model)
    print(f"Test Dataset | accuracy: {accuracy:.2f} - logdet_hF: {logdet_hF:.3f}")
    test_results = {
        "accuracy": accuracy,
    }

    return train_results, test_results

if __name__ == "__main__":
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    args = parse_arguments()
    print()
    print()

    if os.path.exists(os.path.join(args.save, "before_scaling_results.json")):
        print("Finetuned Results already exist. Skipping evaluation.")
        exit(0)

    print("Evaluating fine-tuned models")
    metrics_before_scaling = {}
    for dataset in datasets:
        train_results, test_results = eval_single_task(
            args=args,
            dataset_name=dataset,
            model_path=os.path.join(args.save, f"finetuned_{dataset}.pt"),
        )
        metrics_before_scaling[dataset] = {
            "train": train_results,
            "test": test_results,
        }

    with open(os.path.join(args.save, "before_scaling_results.json"), "w") as f:
        json.dump(metrics_before_scaling, f, indent=4)
