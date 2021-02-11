import torch
import torch.cuda as cuda
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from graphical import visualizer
from sam import sam_wrapper_net

import timm
from pprint import pprint

NUM_EPOCHS = 100

if __name__ == '__main__':
    dev = torch.device("cuda:0" if cuda.is_available() else "cpu")

    TRANSFORM = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    TRAIN_SET = torchvision.datasets.CIFAR10(root='../data', train=True,
                                             download=True, transform=TRANSFORM)
    TRAIN_LOADER = DataLoader(TRAIN_SET, batch_size=4,
                              shuffle=True, num_workers=2)

    TEST_SET = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=TRANSFORM)
    TEST_LOADER = DataLoader(TEST_SET, batch_size=4,
                             shuffle=False, num_workers=2)

    CLASSES = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    m = timm.create_model('mobilenetv3_large_100', pretrained=True)

    wrapper_net = sam_wrapper_net.SAMWrapperNet(dev, model)

    visualizer.show_iterable(wrapper_net.train(TRAIN_LOADER))
