import argparse
import os

import open_clip
import torch
from torch import nn, optim
from torchvision import transforms
from tqdm.auto import tqdm

from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageEncoder
from task_vectors import NonLinearTaskVector

args = argparse.Namespace(model="ViT-B-32", openclip_cachedir=None, cache_dir=None)

base_model = ImageEncoder(args=args)
pretrained_model_path = "./out/pretrained_base.pth"
torch.save(base_model, "./out/pretrained_base.pth")

EPOCHS = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}


def finetune(
    data_location,
    save,
    batch_size,
    lr,
    wd,
    model_name="ViT-B-32",
    openclip_cachedir=None,
    datasets=["MNISTVal", "EuroSAT"],
):
    print(f"Finetuning model with data from {data_location} and saving to {save}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}, Weight decay: {wd}")

    # Define preprocessing transformations
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Ensure consistent size
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),  # Convert PIL.Image to tensor
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalize
        ]
    )

    # Initialize task vectors container
    task_vectors = []

    # Fine-tune the model on each dataset and save after each fine-tuning step
    for dataset_name in datasets:
        print(f"Fine-tuning on dataset: {dataset_name}")
        save_path = os.path.join(save, f"finetuned_{dataset_name}.pt")
        if os.path.exists(save_path):
            task_vector = NonLinearTaskVector(
                pretrained_checkpoint=pretrained_model_path,
                finetuned_checkpoint=save_path,
            )
            task_vectors.append(task_vector)
            continue

        # Load pre-trained model (initial weights)

        model = ImageEncoder(args=args).to("cuda")
        # Load dataset
        train_dataset = get_dataset(dataset_name, preprocess, location=data_location)
        train_loader = get_dataloader(
            train_dataset,
            is_train=True,
            args=argparse.Namespace(batch_size=batch_size, num_workers=2),
        )
        # Initialize optimizer and loss function
        optimizer = optim.SGD(lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()
        # base_model = ImageEncoder(args=args)
        # Training loop
        model.train()
        for epoch in range(EPOCHS[dataset_name]):
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                batch = maybe_dictionarize(batch)
                images, labels = batch["images"].to("cuda"), batch["labels"].to("cuda")

                optimizer.zero_grad()  # Reset gradients
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters

                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        save_path = os.path.join(save, f"finetuned_{dataset_name}.pt")
        torch.save(model, save_path)
        # torch.cuda.empty_cache()
        task_vector = NonLinearTaskVector(
            pretrained_checkpoint=pretrained_model_path, finetuned_checkpoint=save_path
        )
        task_vectors.append(task_vector)  # Store the task vector
        torch.cuda.empty_cache()
        # Save the fine-tuned model after each task

        print(f"Model saved for {dataset_name} at {save_path}")

    # # Combine task vectors by adding them
    # combined_task_vector = None  # Assuming simple sum for task vectors
    # for task_vector in task_vectors:
    #     print(type(task_vector))
    #     if combined_task_vector is None:
    #         combined_task_vector = task_vector
    #         continue
    #     combined_task_vector += task_vector
    # # Apply the combined task vector to the pre-trained model
    # task_vector_instance = NonLinearTaskVector(
    #     vector=combined_task_vector.vector
    # )  # Assuming combined_task_vector is prepared
    # applied_model = task_vector_instance.apply_to(
    #     pretrained_checkpoint=pretrained_model_path
    # )  # Apply the task vector to the model

    # # Save the final model after task addition
    # save_path = os.path.join(save, "finetuned_multitask_model.pt")
    # torch.save(applied_model, save_path)
    # print(f"Final combined multi-task model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for arithmetic tasks."
    )
    parser.add_argument(
        "--data-location", type=str, required=True, help="Path to the datasets"
    )
    parser.add_argument(
        "--save", type=str, required=True, help="Path to save the model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")

    args = parser.parse_args()

    finetune(
        args.data_location,
        args.save,
        args.batch_size,
        args.lr,
        args.wd,
        datasets=["MNISTVal", "EuroSAT", "GTSRB", "RESISC45", "SVHN", "CIFAR10"],
    )


if __name__ == "__main__":
    main()
