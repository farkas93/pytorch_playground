import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from config import playground_config


def main():
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root=playground_config['data_folder'],
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 1

    # Create data loaders.
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    loaded_model = playground_config['model_to_use']
    loaded_model.load_state_dict(torch.load(playground_config['save_location']))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    loaded_model.eval()
    correct = 0
    for X, y in test_dataloader: 
        with torch.no_grad():
            pred = loaded_model(X)
            predicted, actual = classes[pred[0].argmax(0)], classes[y.item()]
            if predicted == actual:
                correct += 1
    
    print("Correct {}/{}".format(correct, len(test_data)))



if __name__ == "__main__":
    main()