import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class WrapperNet(nn.Module):
    def __init__(self, device):
        super(WrapperNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, train_loader, epochs):
        result = []
        for i, data in enumerate(train_loader, 0):
            for e in range(epochs):  # loop over the dataset multiple times
                running_loss = 0.0

                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
            result.append(running_loss)
            running_loss = 0.0
        return result

    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                print(data[0], data[1])
                images, labels = data[0].to(self.device), data[1]#.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total
