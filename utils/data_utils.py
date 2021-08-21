import logging
from paddle.vision import transforms, datasets
from paddle.io import DataLoader

def get_loader(args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.Cifar10(mode="train",
                                    download=True,
                                    transform=transform_train)
        testset = datasets.Cifar10(mode="test",
                                   download=True,
                                   transform=transform_test)

    else:
        trainset = datasets.Cifar100(mode="train",
                                     download=True,
                                     transform=transform_train)
        testset = datasets.Cifar100(mode="test",
                                    download=True,
                                    transform=transform_test)

    train_loader = DataLoader(trainset,
                              batch_size=args.train_batch_size,
                              num_workers=1,
                              use_shared_memory=True)
    test_loader = DataLoader(testset,
                             batch_size=args.eval_batch_size,
                             num_workers=1,
                             use_shared_memory=True)

    return train_loader, test_loader
