from datetime import datetime

import torch
import torch.cuda as cuda
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

import timm
import matplotlib.pyplot as plt

import ensembles.ensemble_net as wr

NUM_EPOCHS = 100
BATCH_SIZE = 100

if __name__ == '__main__':

    t = datetime.now()
    dev = torch.device("cuda:0" if cuda.is_available() else "cpu")

    TRANSFORM = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    TRAIN_SET = torchvision.datasets.CIFAR10(root='../data', train=True,
                                             download=True, transform=TRANSFORM)
    #Only for testing
    ts1 = Subset(TRAIN_SET, range(0, len(TRAIN_SET), 100))
    TRAIN_LOADER = DataLoader(ts1, batch_size=BATCH_SIZE,
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
    wrapper_net = wr.EnsembleNet([model1, model2, model3], NUM_EPOCHS, dev)
    losses = wrapper_net.under_train(TRAIN_LOADER)
    print('Accuracy: ', wrapper_net.test(TEST_LOADER))
    print(losses)
    #plt.plot([i for i, _ in losses], [e for _, e in losses])
    #plt.show()

    print((datetime.now()-t).total_seconds())
