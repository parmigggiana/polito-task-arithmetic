from args import parse_arguments


def eval(data_location, save):
    print(
        f"Evaluating single task model with data from {data_location} and saving to {save}"
    )
    print()


def main():
    args = parse_arguments()
    eval(args.data_location, args.save)


if __name__ == "__main__":
    main()
