import torch
import torch.cuda as cuda
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from graphical import visualizer
import basic.wrapper_net as wr

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

    # CLASSES = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    wrapper_net = wr.WrapperNet(dev)

    wrapper_net.train(TRAIN_LOADER, 1)

    # PATH = './cifar_net.pth'
    # torch.save(wrapper_net.state_dict(), PATH)
    #
    # net = wr.WrapperNet('cpu')
    # net.load_state_dict(torch.load(PATH))

    print('Accuracy: ', wrapper_net.test(TEST_SET))



