import torch
from torchvision import transforms
from tqdm import tqdm

from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from utils import train_diag_fim_logtr

samples_nr = 2000  # How many per-example gradients to accumulate


def eval_single_task(args, dataset_name, model_path):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ImageEncoder(args=args)  # Customize args as needed
    head = get_classification_head(args, dataset_name + "Val")
    model = ImageClassifier(encoder, head)  # Build full model
    if model_path is not None:
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )  # Load the model
    model.eval()  # Set the model to evaluation mode
    # Get the dataset
    dataset = get_dataset(
        dataset_name,
        preprocess=model.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    loader = get_dataloader(
        dataset,
        is_train=False,
        args=args,
    )

    # Initialize variables for evaluation
    correct = 0
    total = 0

    # Start evaluation loop
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Evaluating {dataset_name}")
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
    logdet_hF = train_diag_fim_logtr(args, model, dataset_name, samples_nr)
    print(f"Dataset: {accuracy:.2f}x - logdet_hF: {logdet_hF:.3f}")


if __name__ == "__main__":
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    args = parse_arguments()
    print()
    print("Evaluating pretrained model")
    for dataset in datasets:
        eval_single_task(args=args, dataset_name=dataset, model_path=None)
    print()
    print("Evaluating fine-tuned models")
    for dataset in datasets:
        eval_single_task(
            args=args, dataset_name=dataset, model_path=f"finetuned_{dataset}.pt"
        )
