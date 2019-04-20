import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    trainvalid = datasets.SVHN('data/svhn', split='train', download=True, transform=transform)

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = torch.utils.data.dataset.random_split(trainvalid, [trainset_size, len(trainvalid) - trainset_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True , num_workers=2)
    validloader = DataLoader(validset, batch_size=batch_size)
    testloader = DataLoader(datasets.SVHN('data/svhn', split='test', 
                                           download=True, transform=transform), batch_size=batch_size)

    return trainloader, validloader, testloader