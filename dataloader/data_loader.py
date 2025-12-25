import torch
import torchvision
import torchvision.transforms as transforms


class DatasetLoader:

    def __init__(self, name="cifar10", batch_size= 4):

        self.batch_size = batch_size
        self._load(name)

    def _load(self, name= "cifar10"):

        transform = transforms.Compose(
           [transforms.Resize(32 + int(.25*32)),  # args.img_size + 1/4 *args.img_size
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        if name == "cifar10":
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
            
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)




        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=2)

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                shuffle=False, num_workers=2)

    def get_loader(self, name="train"):

        if name == "train":
            return self.trainloader
        else:
            return self.testloader