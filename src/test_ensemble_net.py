import torch
import torch.cuda as cuda
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optimal
import timm
import matplotlib.pyplot as plt

import ensembles.ensemble_net as wr

NUM_EPOCHS = 1
BATCH_SIZE = 4

if __name__ == '__main__':
    dev = torch.device("cuda:0" if cuda.is_available() else "cpu")

    TRANSFORM = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    TRAIN_SET = torchvision.datasets.CIFAR10(root='../data', train=True,
                                             download=True, transform=TRANSFORM)
    TRAIN_LOADER = DataLoader(TRAIN_SET, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)

    TEST_SET = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=TRANSFORM)
    TEST_LOADER = DataLoader(TEST_SET, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    # CLASSES = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #timm models for ImageNet
    model1 = timm.create_model('tf_efficientnet_b8', pretrained=True)
    model2 = timm.create_model('resnet50', pretrained=True)
    model3 = timm.create_model('tv_resnet34', pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimal.SGD(model1.parameters(), lr=0.001, momentum=0.9)
    wrapper_net = wr.EnsembleNet([model1, model2, model3], criterion, optimizer, 2, dev)
    losses = wrapper_net.under_train(TRAIN_LOADER)
    print('Accuracy: ', wrapper_net.test(TEST_LOADER))
    plt.plot(losses)
