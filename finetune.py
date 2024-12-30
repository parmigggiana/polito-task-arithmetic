import argparse


def finetune(data_location, save, batch_size, lr, wd):
    print(f"Finetuning model with data from {data_location} and saving to {save}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {wd}")
    print()


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
