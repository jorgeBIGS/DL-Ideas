import torchvision
import torchvision.transforms as transforms
import net
import visualizer
from torch.utils.data.dataloader import DataLoader
import torch.cuda as cuda
import torch
 


if __name__=='__main__':
    
    dev = torch.device("cuda:0" if cuda.is_available() else "cpu")
    
    TRANSFORM = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    
    TRAIN_SET = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=TRANSFORM)
    TRAIN_LOADER = DataLoader(TRAIN_SET, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    TEST_SET = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=TRANSFORM)
    TEST_LOADER = DataLoader(TEST_SET, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    CLASSES = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
    net = net.Net()
    
    #Si quito esto me funciona porque lo ejecuta en la cpu
    net.to(dev)
    
    visualizer.show_iterable(net.train(TRAIN_LOADER))
    
