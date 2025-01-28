import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import os
from modeling import ImageEncoder
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from args import parse_arguments

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct

def eval_single_task(model_path, data_location, batch_size=32, device=None, dataset_name="MNIST"):
    """
    Evaluates a fine-tuned model on a single dataset.

    Args:
    - model_path: Path to the fine-tuned model.
    - data_location: Path to the dataset directory.
    - batch_size: Batch size for evaluation.
    - device: The device to run the model on ('cuda' or 'cpu').
    - dataset_name: The name of the dataset to evaluate on.

    Returns:
    - None
    """
    # Set the device
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load the dataset
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the model input size
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Get the dataset
    dataset = get_dataset(dataset_name, preprocess, location=data_location)
    test_loader = get_dataloader(dataset, is_train=False, args=argparse.Namespace(batch_size=batch_size, num_workers=2))
    args = argparse.Namespace(
            model="ViT-B-32",
            openclip_cachedir=None,
            cache_dir=None
        )

    # Initialize the model
    model = ImageEncoder(args=args)  # Customize args as needed
    model.load_state_dict(torch.load(model_path).state_dict())  # Load the fine-tuned model
    model.to(device)  # Move the model to the correct device
    model.eval()  # Set the model to evaluation mode

    # Initialize variables for evaluation
    correct = 0
    total = 0

    # Start evaluation loop
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

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
    print(f"Final accuracy on {dataset_name} dataset: {accuracy:.2f}%")
# Example usage

def main():
    args = parse_arguments()
    eval(args.data_location, args.save)


if __name__ == "__main__":
    main()
