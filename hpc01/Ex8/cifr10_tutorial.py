
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time
from argparse import ArgumentParser

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_n_test(device, batch_size = 4, lr = 0.001, nr_epochs = 2):

    # 1. Load and normalize CIFAR10
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 2. Define a Convolutional Neural Network
    net = Net()
    net.to(device)

    # 3. Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train_results = []

    # Track the ratio of weight updates over weight magnitudes
    track_ratio = True
    if track_ratio:
        initial_weights = {name: param.clone().detach() for name, param in net.named_parameters()}

    # 4. Train the network
    train_start = time()
    for epoch in range(nr_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # Calculate the ratio of weight updates over weight magnitudes
            if track_ratio and i == 0:
                current_weights = {name: param.clone().detach() for name, param in net.named_parameters()}
                max_name_length = max(len(name) for name in initial_weights.keys())
                print(f"Epoch: {epoch} - weight update ratios:")
                for name in initial_weights:
                    weight_update = current_weights[name] - initial_weights[name]
                    weight_magnitude = torch.norm(current_weights[name])
                    update_magnitude_ratio = torch.norm(weight_update) / weight_magnitude
                    print(f"{name.ljust(max_name_length)}: {update_magnitude_ratio.item():.5f}")

            if i % (int(len(trainloader) * 0.16)) == (int(len(trainloader) * 0.16) - 1):
                # print(
                #     f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}"
                # )
                train_results = train_results + [(batch_size, lr, epoch+1, i+1, running_loss/int(len(trainloader) * 0.16))]
                running_loss = 0.0

    train_end = time()

    for res in train_results:
        print("TRA,{},{},{},{},{},{:.3f}".format(res[0], res[1], res[2], nr_epochs, res[3], res[4]))

    print("TRA_TIM,{},{},{},{:.2f}".format(batch_size, lr, nr_epochs, train_end - train_start))

    # 5. Test the network on the test data
    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    test_start = time()
    with torch.no_grad():
        for data in testloader:

            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_end = time()
    print("TST,{},{},{},{:.2f},{:.2f}".format(batch_size, lr, nr_epochs, 100*correct / total, test_end - test_start))

if __name__ == "__main__":
    parser = ArgumentParser(description='Train and test a neural network on CIFAR-10.')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for training if available')
    args = parser.parse_args()

    pos_learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    pos_batch_sizes    = [4, 10, 64, 256]
    pos_epochs    = [2, 4, 6]

    device = torch.device('cuda:0' if args.cuda else 'cpu')

    # for batch_size in pos_batch_sizes:
    #     for lr in pos_learning_rates:
    #         for epochs in pos_epochs:
    #             train_n_test(
    #                 device = device,
    #                 batch_size = batch_size,
    #                 lr=lr,
    #                 nr_epochs = epochs)

    train_n_test(
        device = device,
        batch_size = 64,
        lr=0.01,
        nr_epochs = 10)
