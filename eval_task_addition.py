import argparse

from args import parse_arguments


def eval(args):
    print(
        f"Evaluating task addition model with data from {args.data_location} and saving to {args.save}"
    )
    print()


def main():
    args = parse_arguments()
    eval(args)


if __name__ == "__main__":
    main()
