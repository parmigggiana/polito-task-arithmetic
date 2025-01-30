import os

import torch
from torch import nn, optim
from torchvision import transforms
from tqdm.auto import tqdm

from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder

EPOCHS = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}


def finetune(args, datasets):
    print(
        f"Finetuning model with data from {args.data_location} and saving to {args.save}"
    )
    print(
        f"Batch size: {args.batch_size}, Learning rate: {args.lr}, Weight decay: {args.wd}"
    )

    # Define preprocessing transformations
    # preprocess = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),  # Ensure consistent size
    #         transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    #         transforms.ToTensor(),  # Convert PIL.Image to tensor
    #         transforms.Normalize(
    #             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    #         ),  # Normalize
    #     ]
    # )
    encoder = ImageEncoder(args).to("cuda")
    # Fine-tune the model on each dataset and save after each fine-tuning step
    for dataset_name in datasets:
        save_path = os.path.join(args.save, f"finetuned_{dataset_name}.pt")
        if os.path.exists(save_path):
            continue
        print(f"Fine-tuning on dataset: {dataset_name}")
        head = get_classification_head(args, dataset_name + "Val")
        model = ImageClassifier(encoder, head)  # Build full model
        # model.to("cuda")
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
        # base_model = ImageEncoder(args=args)
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

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(loader)}")

        save_path = os.path.join(args.save, f"finetuned_{dataset_name}.pt")
        torch.save(model, save_path)
        torch.cuda.empty_cache()
        # Save the fine-tuned model after each task

        print(f"Model saved for {dataset_name} at {save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    finetune(
        args,
        datasets=["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"],
    )
