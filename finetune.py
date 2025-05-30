import json
import os

import torch
from torch import nn, optim
from tqdm.auto import tqdm

from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from utils import train_diag_fim_logtr

EPOCHS = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}

samples_nr = 200


def eval(args, loader, dataset_name, model):
    # Initialize variables for evaluation
    correct = 0
    total = 0

    # Start evaluation loop
    with torch.no_grad():
        # progress_bar = tqdm(loader, desc=f"Evaluating {dataset_name}")
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Update the count of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Optionally update the progress bar with accuracy
            # accuracy = 100 * correct / total
            # progress_bar.set_postfix({"accuracy": accuracy})

    # Final accuracy
    accuracy = 100 * correct / total
    logdet_hF = train_diag_fim_logtr(args, model, dataset_name, samples_nr)

    return accuracy, logdet_hF


def eval_acc(args, loader, model):
    # Initialize variables for evaluation
    correct = 0
    total = 0

    # Start evaluation loop
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Update the count of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Final accuracy
    accuracy = 100 * correct / total

    return accuracy


def finetune(args, datasets):
    print(
        f"Finetuning model with data from {args.data_location} and saving to {args.save}"
    )
    print(
        f"Batch size: {args.batch_size}, Learning rate: {args.lr}, Weight decay: {args.wd}"
    )

    encoder = ImageEncoder(args).to("cuda")

    ckpt_path = os.path.join(args.save, f"base.pt")
    encoder.save(ckpt_path)  # Save the base model

    # Fine-tune the model on each dataset and save after each fine-tuning step
    pre_trained_metrics = {}
    for dataset_name in datasets:
        save_path = os.path.join(args.save, f"finetuned_{dataset_name}.pt")
        if os.path.exists(save_path):
            continue

        head = get_classification_head(args, dataset_name + "Val")
        model = ImageClassifier(encoder, head).to(
            "cuda"
        )  # Build full model and move to GPU

        model.freeze_head()  # Freeze the classification head
        dataset = get_dataset(
            dataset_name + "Val",
            preprocess=model.train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        loader = get_dataloader(dataset, is_train=True, args=args)

        # Initialize optimizer and loss function
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss()

        print(f"Fine-tuning on dataset: {dataset_name}")
        # Pre-trained model metrics
        model.eval()
        pre_acc, pre_logdet = eval(args, loader, dataset_name, model)
        pre_trained_metrics[dataset_name] = {
            "train": {
                "accuracy": pre_acc,
                "logdet_hF": pre_logdet,
            }
        }
        test_dataset = get_dataset(
            dataset_name,
            preprocess=model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        test_loader = get_dataloader(test_dataset, is_train=False, args=args)
        pre_acc = eval_acc(args, test_loader, model)
        pre_trained_metrics[dataset_name]["test"] = {
            "accuracy": pre_acc,
        }

        # Training loop
        model.train()
        for epoch in range(EPOCHS[dataset_name]):
            running_loss = 0.0
            progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                data = maybe_dictionarize(batch)
                images, labels = data["images"].to("cuda"), data["labels"].to("cuda")

                optimizer.zero_grad()  # Reset gradients
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters

                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

            # print(f"Epoch {epoch + 1}, Loss: {running_loss / len(loader)}")

        save_path = os.path.join(args.save, f"finetuned_{dataset_name}.pt")

        model.image_encoder.save(save_path)
        torch.cuda.empty_cache()
        print()

    if pre_trained_metrics != {}:
        with open(os.path.join(args.save, "pre_trained_results.json"), "w") as f:
            json.dump(pre_trained_metrics, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    finetune(
        args,
        datasets=["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"],
    )
