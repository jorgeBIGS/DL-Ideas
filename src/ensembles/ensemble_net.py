import torch
import torch.nn as nn


class EnsembleNet(nn.Module):
    def __init__(self, models: list, criterion, optimizer, epochs, device):
        super(EnsembleNet, self).__init__()
        self.models = models
        total = 0
        for model in models:
            model.to(device)
            total += [param.nelement() for param in model.parameters()][-1]
            for param in model.parameters():
                param.requires_grad_(False)

        self.relu = nn.ReLU(total)
        self.fc1 = nn.Linear(in_features=total, out_features=10)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.to(device)

    def forward(self, x):
        results = [model(x.clone()) for model in self.models]
        x = torch.flatten(torch.tensor(results))
        #x = results[0]  # clone to make sure x is not changed by inplace methods
        #x = x.view(x.size(0), -1)

        #for i in range(1, len(results)):
        #    x2 = results[i]
        #    x2 = x2.view(x2.size(0), -1)
        #    x = torch.cat((x, x2), dim=1)

        x = self.relu(x)
        return self.fc1(x)

    def under_train(self, train_loader):
        result = []
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                result += [loss.item()]
                self.optimizer.step()

        return result

    def test(self, test_loader):
        self.train()
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # calculate outputs by running images through the network
                images = images.to(self.device)
                outputs = self(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to(torch.device('cpu'))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
