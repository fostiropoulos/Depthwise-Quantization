import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import glob
from PIL import Image
import pickle

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


class ImageFolderUnlabeled(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample


class ImageFolderAnon(Dataset):
    def __init__(self, root_dir, transform=None):
        types = (
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.PNG",
            "*.JPEG",
            "*.JPG",
        )

        self.root_dir = root_dir
        self.transform = transform
        files = []
        for ftype in types:
            pattern = os.path.join(self.root_dir, ftype)
            files.extend(glob.glob(pattern))
        self.files = files
        self.class_to_idx = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.files[idx])
        if self.transform:
            img = self.transform(img)

        return img


def unpickle(file):
    with open(file, "rb") as fo:
        dic = pickle.load(fo)
    return dic


class ImageNet(Dataset):
    def __init__(self, root_dir, transform=None, test=False):
        self.x = []
        self.y = []
        self.img_size = 32
        for name in os.listdir(root_dir):
            tmpp = os.path.join(root_dir, name)
            d = unpickle(tmpp)
            tmpx = torch.from_numpy(d["data"])
            self.x.append(tmpx)
            self.y.append(torch.LongTensor(d["labels"]))
        if not test:
            self.x = torch.cat(self.x, 0)
            self.y = torch.cat(self.y, 0)
        if test:
            self.x = self.x[0]
            self.y = self.y[0]
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.x[idx], self.y[idx] - 1
        img = img.reshape((self.img_size, self.img_size, 3))
        img = Image.fromarray(img.numpy())
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.y.shape[0]


def cifar10_transform(image_size, augment, normalize=True):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size) if augment else lambda x: x,
            transforms.RandomHorizontalFlip(p=0.2) if augment else lambda x: x,
            transforms.ColorJitter(
                brightness=0.35, contrast=0.5, saturation=0.3, hue=0.03
            )
            if augment
            else lambda x: x,
            transforms.RandomVerticalFlip(p=0.05) if augment else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD) if normalize else lambda x: x,
        ]
    )


def imagenet_transform(image_size, augment):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip() if augment else lambda x: x,
            transforms.RandomResizedCrop(image_size)
            if augment
            else transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
    )


def load_dataset(
    dataset,
    image_size,
    batch_size=32,
    drop_last=True,
    shuffle=True,
    augment=True,
    normalize=True,
    label=False,
):
    from torch.utils.data import DataLoader

    if dataset == "cifar10":
        transform = cifar10_transform(image_size, augment, normalize)
        test_transform = cifar10_transform(image_size, False)
    else:
        transform = imagenet_transform(image_size, augment)
        test_transform = imagenet_transform(image_size, False)

    if dataset == "cifar10":
        dl_path = "/tmp/cifar10"
        is_exist = not os.path.exists(dl_path)
        trainset = datasets.CIFAR10(
            dl_path, train=True, transform=transform, download=is_exist
        )
        testset = datasets.CIFAR10(
            dl_path, train=False, transform=test_transform, download=is_exist
        )
    elif dataset == "imageNet32":
        train_path = os.path.join(dataset, "train")
        val_path = os.path.join(dataset, "val")
        testset = ImageNet(val_path, transform=test_transform, test=True)
        trainset = ImageNet(train_path, transform=transform)

    else:
        train_path = os.path.join(dataset, "train")
        val_path = os.path.join(dataset, "val")
        train_folders = os.path.join(train_path, "*")
        train_glob = glob.glob(train_folders)

        if len(train_glob) > 0:
            if not label:
                trainset = ImageFolderUnlabeled(train_path, transform=transform)
            if label:
                trainset = datasets.ImageFolder(train_path, transform=transform)
            if os.path.exists(val_path):
                if not label:
                    testset = ImageFolderUnlabeled(val_path, transform=test_transform)
                if label:
                    testset = datasets.ImageFolder(val_path, transform=test_transform)
            else:
                testset = None
        else:
            trainset = ImageFolderAnon(dataset, transform=transform)
            print("Dataset length: %s" % len(trainset))
            testset = None

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
    )
    if testset:
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=True, num_workers=0
        )
    else:
        testloader = None
    return trainloader, testloader


def get_device_name(tensor):

    device = tensor.get_device()
    device = "cpu" if device == -1 else device
    return device


def denorm_batch(img, device=None):
    """
    Normalized batch (B,C,H,W) of images in (mean-std,mean+std) from (0,1) space transformed to [0,255] space
    """

    device = get_device_name(img) if not device else device
    std = torch.tensor(IMG_STD, device=device).view(1, 3, 1, 1)
    mean = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1)
    return (torch.clamp(img * std + mean, 0, 1) * 255).type(torch.long)


def norm_batch(img, device=None):
    device = get_device_name(img) if not device else device
    std = torch.tensor(IMG_STD, device=device).view(1, 3, 1, 1)
    mean = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1)

    if img.shape[1] == 1:
        (torch.clamp(img, 0, 1) * 255).type(torch.long)
    return (img) / mean - std

