import torch


def main():

    print("CUDA is available?: ", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())


if __name__ == "__main__":
    main()
