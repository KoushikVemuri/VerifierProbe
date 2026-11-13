"""train_mnist.py -- train a small MNIST classifier (CPU) and save it.

    python train_mnist.py

The weights file mnist_fc.pt ships with the package. The result in the
README depends on those exact weights — retraining gives slightly
different ones.
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms


def build_model() -> nn.Sequential:
    """Small fully connected network. Verifiers get slow on big networks."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 10),
    )


def main() -> None:
    """Train for 5 epochs, print test accuracy, save mnist_fc.pt."""
    torch.manual_seed(0)
    to_tensor = transforms.ToTensor()  # keeps pixel values in [0, 1]
    train = datasets.MNIST("data", train=True, download=True, transform=to_tensor)
    test = datasets.MNIST("data", train=False, download=True, transform=to_tensor)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss_fn(model(images), labels).backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                correct += (model(images).argmax(1) == labels).sum().item()
        print(f"epoch {epoch + 1}: test accuracy {correct / len(test):.4f}")

    torch.save(model.state_dict(), "mnist_fc.pt")
    print("saved mnist_fc.pt")


if __name__ == "__main__":
    main()