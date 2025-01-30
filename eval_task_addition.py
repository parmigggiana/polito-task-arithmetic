import argparse

import torch
from torchvision import transforms
from tqdm import tqdm

from datasets.common import get_dataloader
from datasets.registry import get_dataset
from task_vectors import NonLinearTaskVector


def eval_task_addition(pretrained_model_path="./out/pretrained_base.pth", data_location="./data", batch_size=32,
                       device="cuda",
                       datasets=None):
    """
    Evaluates a model with applied task vectors across multiple datasets.
    """
    # Combine task vectors by adding them
    if datasets is None:
        datasets = ["MNISTVal", "EuroSAT", "GTSRB", "RESISC45", "SVHN", "DTD"]
    combined_task_vector = None  # Assuming simple sum for task vectors
    task_vectors = []
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    for dataset_name in datasets:
        print(f"Creating task vector for {dataset_name}...")

        finetuned_model_path = f"./out/finetuned_{dataset_name}.pt"
        task_vector = NonLinearTaskVector(pretrained_checkpoint=pretrained_model_path,
                                          finetuned_checkpoint=finetuned_model_path)

        task_vectors.append(task_vector)  # Store the task vector

    for task_vector in task_vectors:
        print(type(task_vector))
        if combined_task_vector is None:
            combined_task_vector = task_vector
            continue
        combined_task_vector += task_vector

    task_addition_model = combined_task_vector.apply_to(pretrained_model_path).to(device).eval()
    print("Task vector applied. Evaluating on multiple datasets...")

    results = {}
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the model input size
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    for dataset_name in datasets:
        print(f"Evaluating on {dataset_name}...")
        dataset = get_dataset(dataset_name, preprocess, location=data_location)
        test_loader = get_dataloader(dataset, is_train=False,
                                     args=argparse.Namespace(batch_size=batch_size, num_workers=2))

        correct = 0
        total = 0
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc=f"Testing {dataset_name}")
            for batch in progress_bar:
                batch = batch if isinstance(batch, dict) else {"images": batch[0], "labels": batch[1]}
                images, labels = batch["images"].to(device), batch["labels"].to(device)

                outputs = task_addition_model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                accuracy = 100 * correct / total
                progress_bar.set_postfix({"accuracy": accuracy})

        results[dataset_name] = 100 * correct / total

    print("Evaluation completed. Results:")
    for dataset, acc in results.items():
        print(f"{dataset}: {acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate task-added model across multiple datasets.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--task-vector-path", type=str, required=True, help="Path to the task vector.")
    parser.add_argument("--data-location", type=str, required=True, help="Path to datasets.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run evaluation.")
    parser.add_argument("--datasets", nargs='+',
                        default=["MNISTVal", "EuroSAT", "GTSRB", "RESISC45", "SVHN", "DTD"],
                        help="Datasets to evaluate on.")

    args = parser.parse_args()
    eval_task_addition(args.model_path, args.data_location, args.batch_size, args.device,
                       args.datasets)


if __name__ == "__main__":
    main()
