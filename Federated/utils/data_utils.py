from math import log
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb


def get_auxiliary_data(config, transforms, dataset, n_classes, n_aux):
    # output an auxiliary dataset with n_aux sample per class
    # the output is a list of tuples (data, label)
    data = dataset.data
    label = dataset.targets

    aux_data = []
    for c in range(n_classes):
        mask_c = label == c
        data_c = data[mask_c.numpy()]
        label_c = label[mask_c]
        assert(len(data_c) > n_aux)
        aux_data_c = data_c[:n_aux]
        aux_label_c = label_c[:n_aux]

        aux_data_c = torch.stack([transforms(aux_data_c[i]) for i in range(n_aux)])
        # if config.dataset == "cifar10" or config.dataset == "cifar100":
        #     aux_data_c = aux_data_c.permute(0, 3, 1, 2)
        aux_data.append((aux_data_c, aux_label_c))

    return aux_data

def create_imbalance(dataset, reduce_classes=(0,), reduce_to_ratio=.2):
    data = dataset.data
    label = dataset.targets

    reduce_mask = torch.zeros(data.shape[0], dtype=torch.bool)
    for reduce_class in reduce_classes:
        reduce_mask = torch.logical_or(reduce_mask, label == reduce_class)
    preserve_mask = torch.logical_not(reduce_mask)

    label_reduce = label[reduce_mask]
    len_reduce = label_reduce.shape[0]
    label_reduce = label_reduce[:max(1, int(len_reduce * reduce_to_ratio))]
    label_preserve = label[preserve_mask]

    label = torch.cat([label_reduce, label_preserve], dim=0)

    preserve_mask_np = preserve_mask.numpy()
    reduce_mask_np = reduce_mask.numpy()

    data_reduce = data[reduce_mask_np]
    data_reduce = data_reduce[:max(1, int(len_reduce * reduce_to_ratio))]
    data_preserve = data[preserve_mask_np]

    data = np.concatenate([data_reduce, data_preserve], axis=0)

    remain_len = label.shape[0]

    rand_index = torch.randperm(remain_len)
    rand_index_np = rand_index.numpy()

    dataset.data = data[rand_index_np]
    dataset.targets = label[rand_index]

    return dataset


def load_dataset(args):
    if args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(root='datasets/' + args.dataset, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.CIFAR10(root='datasets/' + args.dataset, train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 3
        img_size = 32
    elif args.dataset == "cifar100":
        dataset_train = datasets.CIFAR100(root='datasets/' + args.dataset, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.CIFAR100(root='datasets/' + args.dataset, train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 100
        n_channels = 3
        img_size = 32
    elif args.dataset == "mnist":
        dataset_train = datasets.MNIST(root='datasets/' + args.dataset, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.MNIST(root='datasets/' + args.dataset, train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 1
        img_size = 28
    elif args.dataset == "fashion-mnist":
        dataset_train = datasets.FashionMNIST(root='datasets/' + args.dataset, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.FashionMNIST(root='datasets/' + args.dataset, train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 1
        img_size = 28
    elif args.dataset == "emnist":
        dataset_train = datasets.EMNIST(root='datasets/' + args.dataset, split="letters", download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets)) - 1
        dataset_test = datasets.EMNIST(root='datasets/' + args.dataset, split="letters", train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets)) - 1
        n_classes = 26
        n_channels = 1
        img_size = 28
    else:
        raise NotImplementedError

    return dataset_train, dataset_test, n_classes, n_channels, img_size

def split_dataset(args, dataset: VisionDataset, transform=None, ratio_per_client=None, test_sz=.1, reduce_classes=[0, 2, 4]):
    data = dataset.data
    data = data.numpy() if torch.is_tensor(data) is True else data
    label = dataset.targets
    n_workers = args.n_workers
    homo_ratio = args.homo_ratio

    # centralized case, no need to split
    if n_workers == 1:
        return [make_dataset(data, label, dataset.train, transform)]

    if ratio_per_client is None:
        if args.heterogeneity == 'mix':
            n_data = data.shape[0]

            n_homo_data = int(n_data * homo_ratio)

            n_homo_data = n_homo_data - n_homo_data % n_workers
            n_data = n_data - n_data % n_workers

            if n_homo_data > 0:
                data_homo, label_homo = data[0:n_homo_data], label[0:n_homo_data]
                data_homo_list, label_homo_list = np.split(data_homo, n_workers), label_homo.chunk(n_workers)

            if n_homo_data < n_data:
                data_hetero, label_hetero = data[n_homo_data:n_data], label[n_homo_data:n_data]
                label_hetero_sorted, index = torch.sort(label_hetero)
                data_hetero_sorted = data_hetero[index]

                data_hetero_list, label_hetero_list = np.split(data_hetero_sorted, n_workers), label_hetero_sorted.chunk(
                    n_workers)

            if 0 < n_homo_data < n_data:
                data_list = [np.concatenate([data_homo, data_hetero], axis=0) for data_homo, data_hetero in
                            zip(data_homo_list, data_hetero_list)]
                label_list = [torch.cat([label_homo, label_hetero], dim=0) for label_homo, label_hetero in
                            zip(label_homo_list, label_hetero_list)]
            elif n_homo_data < n_data:
                data_list = data_hetero_list
                label_list = label_hetero_list
            else:
                data_list = data_homo_list
                label_list = label_homo_list

        elif args.heterogeneity == 'dir':
            n_cls = (int(torch.max(label))) + 1
            n_data = data.shape[0]

            cls_priors = np.random.dirichlet(alpha=[args.dir_level] * n_cls, size=n_workers)

            # cls_priors_init = cls_priors # Used for verification
            prior_cumsum = np.cumsum(cls_priors, axis=1)
            idx_list = [np.where(label == i)[0] for i in range(n_cls)]
            cls_amount = [len(idx_list[i]) for i in range(n_cls)]
            idx_worker = [[None] for i in range(n_workers)]

            for curr_worker in range(n_workers):
                for data_sample in range(n_data // n_workers):
                    curr_prior = prior_cumsum[curr_worker]
                    cls_label = np.argmax(np.random.uniform() <= curr_prior)
                    while cls_amount[cls_label] <= 0:
                        # If you run out of samples
                        correction = [[1 - cls_priors[i, cls_label]] * n_cls for i in range(n_workers)]
                        cls_priors = cls_priors / correction
                        cls_priors[:, cls_label] = [0] * n_workers
                        curr_prior = np.cumsum(cls_priors, axis=1)
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)

                    cls_amount[cls_label] -= 1
                    if idx_worker[curr_worker] == [None]:
                        idx_worker[curr_worker] = [idx_list[cls_label][0]]
                    else:
                        idx_worker[curr_worker] = idx_worker[curr_worker] + [idx_list[cls_label][0]]

                    idx_list[cls_label] = idx_list[cls_label][1::]
            data_list = [data[idx_worker[curr_worker]] for curr_worker in range(n_workers)]
            label_list = [label[idx_worker[curr_worker]] for curr_worker in range(n_workers)]
        else:
            raise ValueError("heterogeneity should be mix or dir")
    else:
        n_cls = (int(torch.max(label))) + 1
        n_data = data.shape[0]
        cls_priors = ratio_per_client
        # compute number of samples
        data_per_worker = test_sz*n_data*cls_priors/n_cls
        idx_list = [np.where(label == i)[0] for i in range(n_cls)]
        samples_per_class = [len(idxs) for idxs in idx_list]
        # check there's enough data
        assert(min(samples_per_class)>cls_priors.max()*test_sz*n_data/n_cls)
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        idx_worker = [[] for i in range(n_workers)]
        for curr_worker in range(n_workers):
            for cls_idxs, num_samples in zip(idx_list, data_per_worker[curr_worker]):
                # floor num samples
                num_samples = int(num_samples)
                if num_samples>0:
                    # choose randomly
                    chosen_idx = np.random.choice(cls_idxs, size=num_samples, replace=False)
                    idx_worker[curr_worker] = idx_worker[curr_worker]+chosen_idx.tolist()
            assert(len(idx_worker[curr_worker]))          
        data_list = [data[idx_worker[curr_worker]] for curr_worker in range(n_workers)]
        label_list = [label[idx_worker[curr_worker]] for curr_worker in range(n_workers)]
    ##################
    # Log imbalance
    ###################
    n_cls = (int(torch.max(label))) + 1 
    clients_post = []
    for client in range(n_workers):
        label = label_list[client]
        entropy = 0
        cls_post = []
        minority_frac = 0
        for cls in range(n_cls):
            frac = (torch.sum(label==cls)/len(label)).item()
            assert(not np.isnan(frac))
            cls_post.append(frac)
            if ratio_per_client is None:
                wandb.log({f"frac_samples/client_{client}/class_{cls}/train":frac})
            else:
                wandb.log({f"frac_samples/client_{client}/class_{cls}/test":frac})
            if frac>0:
                entropy += -frac*log(frac)
            if cls in reduce_classes:
                minority_frac += frac
        assert(np.sum(cls_post)>0)
        if ratio_per_client is None:
            wandb.log({f"entropy/client_{client}/train":entropy})
            wandb.log({f"frac_minority/client_{client}/train":minority_frac})
        else:
            wandb.log({f"entropy/client_{client}/test":entropy})
            wandb.log({f"frac_minority/client_{client}/test":minority_frac})
        clients_post.append(cls_post)

    return [make_dataset(_data, _label, dataset.train, transform) for _data, _label in zip(data_list, label_list)], np.array(clients_post)


class LocalDataset(VisionDataset):
    def __init__(self, data, label, train, transform=None, root: str = ""):
        super().__init__(root, transform)
        self.data = data
        self.label = label
        self.transform = transform
        self.train = train
        assert data.shape[0] == label.shape[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.label[item]


def make_dataset(data, label, train, transform):
    return LocalDataset(data, label, train, transform)




normalize_cifar10 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
normalize_mnist = transforms.Normalize(mean=(0.1307,), std=(0.3081,))


# normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

def make_transforms(args, train=True):
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        if train:
            if not args.no_data_augmentation:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    normalize_cifar10,
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize_cifar10,
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize_cifar10,
            ])
    elif args.dataset == "mnist":
        if train:
            if not args.no_data_augmentation:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    normalize_mnist,
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize_mnist,
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize_mnist,
            ])
    elif args.dataset == "fashion-mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError

    return transform


def make_dataloader(args, dataset: LocalDataset, distributed=True, train=True):
    if train:
        batch_size = dataset.data.shape[0]
    else:
        batch_size=args.test_batch_size
    if distributed:
        batch_size =  batch_size// args.client_step_per_epoch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)
    return dataloader