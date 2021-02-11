import torch.nn as nn
import torch.optim as optim

from sam.sam import SAM


class SAMWrapperNet(nn.Module):
    def __init__(self, device, model, epochs=100, batch_size=500):
        super(SAMWrapperNet, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        base_optimizer = optim.SGD  # define an optimizer for the "sharpness-aware" update
        self.optimizer = SAM(self.parameters(), base_optimizer, lr=0.1, momentum=0.9)
        self.device = device
        self.epochs = epochs
        self.batch = batch_size

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def train(self, train_loader):
        result = []

        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                for input1, output in data:
                    def closure():
                        loss1 = self.criterion(output, self.model(input))
                        loss1.backward()
                        return loss1

                    loss = self.criterion(output, self.model(input1))
                    loss.backward()
                    self.optimizer.step(closure)
                    self.optimizer.zero_grad()
                '''
                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data
                inputs.to(self.device)
                labels.to(self.device)

                # forward + backward + optimize

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    result.append('[%d, %5d] loss: %.3f' %
                                  (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                '''
        return result
