import argparse


def eval(data_location, save):
    print(
        f"Evaluating task addition model with data from {data_location} and saving to {save}"
    )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for arithmetic tasks."
    )
    parser.add_argument(
        "--data-location",
        type=str,
        required=True,
        help="Path to the datasets (e.g., /path/to/datasets/)",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Path to save the model (e.g., /path/to/save/)",
    )

    args = parser.parse_args()

    eval(args.data_location, args.save)


if __name__ == "__main__":
    main()
