
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time

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

def load(batch_size = 4):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    tr_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    tst_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return tr_loader, tst_loader

if __name__ == "__main__":

    # 1. Define a Convolutional Neural Network
    net = Net()

    # 2. Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    train_results = []
    # train_times = []
    # test_results = []
    pos_learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    pos_batch_sizes    = [4, 10, 64, 1024]

    for batch_size in pos_batch_sizes:
        # 3. Load and normalize CIFAR10
        trainloader,testloader = load(batch_size)

        for lr in pos_learning_rates:
            train_results = []

            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

            # 4. Train the network
            train_start = time()
            for epoch in range(2):  # loop over the dataset multiple times

                running_loss = 0.0

                for i, data in enumerate(trainloader, 0):

                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()

                    if i % 2000 == 1999:  # print every 2000 mini-batches
                        # print("TRA",batch_size, lr, epoch+1, i+1, running_loss/2000, sep=',')
                        train_results = [(batch_size, lr, epoch+1, i+1, running_loss/2000)] + train_results
                        # print(
                        #     f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}"
                        # )

                        running_loss = 0.0

            train_end = time()
            # print("Finished Training, batch_size: {}, learning rate: {} in {:.2f} s".format(batch_size, lr, train_end - train_start))
            for res in train_results:
                print("TRA", res[0], res[1], res[2], res[3], res[4], sep = ',')

            print("TRA_TIM", batch_size, lr, train_end - train_start, sep = ',')
            # train_times = [(batch_size, lr, train_end - train_start)] + train_times

            # 5. Test the network on the test data
            correct = 0
            total = 0

            # since we're not training, we don't need to calculate the gradients for our outputs
            test_start = time()

            with torch.no_grad():
                for data in testloader:

                    images, labels = data

                    # calculate outputs by running images through the network
                    outputs = net(images)

                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            test_end = time()
            print("TST", batch_size, lr, 100*correct / total, test_end - test_start, sep = ',')
            # test_results = [(batch_size, 100*correct / total, test_end - test_start)]

    # print(train_results)
    # print(train_times)
    # print(test_results)

    # print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
    # print("Testing took {:.3f} s".format(test_end - test_start))
