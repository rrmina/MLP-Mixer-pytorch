import torch
from torchvision import datasets, transforms

def get_loaders(args):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.dataset.lower() == 'mnist':
        trainset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
        testset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        image_size, n_image_channel, n_output = 28, 1, 10

    elif args.dataset.lower() == 'cifar10':
        trainset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
        testset = datasets.CIFAR10(root='data', train=False, transform=transform, download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        image_size, n_image_channel, n_output = 32, 3, 10

    elif args.dataset.lower() == 'cifar100':
        trainset = datasets.CIFAR100(root='data', train=True, transform=transform, download=True)
        testset = datasets.CIFAR100(root='data', train=False, transform=transform, download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        image_size, n_image_channel, n_output = 32, 3, 100

    else:
        raise Exception("[ERROR] The dataset " + str(args.dataset) + " is not supported!")

    return trainloader, testloader, image_size, n_image_channel, n_output
