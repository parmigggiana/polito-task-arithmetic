import argparse
import os

import torch
from torch import nn, optim
from torchvision import transforms  # Import transforms
from tqdm.auto import tqdm

from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageEncoder
from utils import torch_save


def finetune(data_location, save, batch_size, lr, wd, model_name="ViT-B-32", openclip_cachedir=None):
    print(f"Finetuning model with data from {data_location} and saving to {save}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}, Weight decay: {wd}")

    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure consistent size
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.ToTensor(),  # Convert PIL.Image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Load dataset
    train_dataset = get_dataset("MNISTVal", preprocess, location=data_location)  # Replace "MNISTVal" with your dataset
    train_loader = get_dataloader(train_dataset, is_train=True,
                                  args=argparse.Namespace(batch_size=batch_size, num_workers=2))

    # Initialize model
    args = argparse.Namespace(
        model=model_name,
        openclip_cachedir=openclip_cachedir,
        cache_dir=None
    )
    model = ImageEncoder(args=args)  # Pass a valid args object
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Training loop
    model.train()
    for epoch in range(10):  # Number of epochs can be adjusted
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            batch = maybe_dictionarize(batch)
            images, labels = batch["images"].to("cuda"), batch["labels"].to("cuda")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (len(progress_bar) + 1))

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Save fine-tuned model
    save_path = os.path.join(save, "finetuned_model.pt")
    torch_save(model, save_path)
    print(f"Model saved to {save_path}")


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

    finetune(args.data_location, args.save, args.batch_size, args.lr, args.wd)


if __name__ == "__main__":
    main()
