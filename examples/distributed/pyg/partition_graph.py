import argparse
import os
import os.path as osp

import torch
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, MovieLens, Reddit
from torch_geometric.distributed import Partitioner
from torch_geometric.utils import mask_to_index


def partition_dataset(
    dataset_name: str,
    root_dir: str,
    num_parts: int,
    recursive: bool = False,
    use_sparse_tensor: bool = False,
):
    if not osp.isabs(root_dir):
        path = osp.dirname(osp.realpath(__file__))
        root_dir = osp.join(path, root_dir)

    dataset_dir = osp.join(root_dir, 'dataset', dataset_name)
    dataset = get_dataset(dataset_name, dataset_dir, use_sparse_tensor)
    data = dataset[0]

    save_dir = osp.join(root_dir, 'partitions', dataset_name,
                        f'{num_parts}-parts')

    partitions_dir = osp.join(save_dir, f'{dataset_name}-partitions')
    partitioner = Partitioner(data, num_parts, partitions_dir, recursive)
    partitioner.generate_partition()

    print('-- Saving label ...')
    label_dir = osp.join(save_dir, f'{dataset_name}-label')
    os.makedirs(label_dir, exist_ok=True)

    split_dim = 0
    if dataset_name == 'ogbn-mag':
        split_data = data['paper']
        label = split_data.y
    else:
        split_data = data
        if dataset_name == 'ogbn-products':
            label = split_data.y.squeeze()
        elif dataset_name == 'Reddit':
            label = split_data.y
        elif dataset_name == 'MovieLens':
            label = split_data[data.edge_types[0]].edge_label
            split_dim = 1

    torch.save(label, osp.join(label_dir, 'label.pt'))

    train_idx, val_idx, test_idx = get_idx_split(dataset, dataset_name,
                                                 split_data)

    print('-- Partitioning training indices ...')
    train_idx = train_idx.split(
        train_idx.size(split_dim) // num_parts, dim=split_dim)
    train_part_dir = osp.join(save_dir, f'{dataset_name}-train-partitions')
    os.makedirs(train_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(train_idx[i], osp.join(train_part_dir, f'partition{i}.pt'))

    print('-- Partitioning validation indices ...')
    val_idx = val_idx.split(
        val_idx.size(split_dim) // num_parts, dim=split_dim)
    val_part_dir = osp.join(save_dir, f'{dataset_name}-val-partitions')
    os.makedirs(val_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(val_idx[i], osp.join(val_part_dir, f'partition{i}.pt'))

    print('-- Partitioning test indices ...')
    test_idx = test_idx.split(
        test_idx.size(split_dim) // num_parts, dim=split_dim)
    test_part_dir = osp.join(save_dir, f'{dataset_name}-test-partitions')
    os.makedirs(test_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(test_idx[i], osp.join(test_part_dir, f'partition{i}.pt'))


def get_dataset(name, dataset_dir, use_sparse_tensor=False):
    transforms = []
    if use_sparse_tensor:
        transforms = [T.ToSparseTensor(remove_edge_index=False)]

    if name == 'ogbn-mag':
        transforms = [T.ToUndirected(merge=True)] + transforms
        return OGB_MAG(
            root=dataset_dir,
            preprocess='metapath2vec',
            transform=T.Compose(transforms),
        )

    elif name == 'ogbn-products':
        transforms = [T.RemoveDuplicatedEdges()] + transforms
        return PygNodePropPredDataset(
            'ogbn-products',
            root=dataset_dir,
            transform=T.Compose(transforms),
        )

    elif name == 'MovieLens':
        transforms = [T.ToUndirected(merge=True)] + transforms
        return MovieLens(
            root=dataset_dir,
            model_name='all-MiniLM-L6-v2',
            transform=T.Compose(transforms),
        )

    elif name == 'Reddit':
        return Reddit(
            root=dataset_dir,
            transform=T.Compose(transforms),
        )


def get_idx_split(dataset, dataset_name, split_data):
    if dataset_name == 'ogbn-mag' or dataset_name == 'Reddit':
        train_idx = mask_to_index(split_data.train_mask)
        test_idx = mask_to_index(split_data.test_mask)
        val_idx = mask_to_index(split_data.val_mask)

    elif dataset_name == 'ogbn-products':
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        test_idx = split_idx['test']
        val_idx = split_idx['valid']

    elif dataset_name == 'MovieLens':
        edge_type = ('user', 'rates', 'movie')
        train_data, val_data, test_data = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            neg_sampling_ratio=0.0,
            edge_types=[edge_type],
            rev_edge_types=[('movie', 'rev_rates', 'user')],
        )(dataset[0])

        train_idx = train_data[edge_type].edge_label_index
        val_idx = val_data[edge_type].edge_label_index
        test_idx = test_data[edge_type].edge_label_index

    return train_idx, val_idx, test_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--dataset', type=str,
        choices=['ogbn-mag', 'ogbn-products', 'MovieLens',
                 'Reddit'], default='ogbn-products')
    add('--root_dir', default='../../../data', type=str)
    add('--num_partitions', type=int, default=2)
    add('--recursive', action='store_true')
    # TODO (kgajdamo) Add support for arguments below:
    # add('--use-sparse-tensor', action='store_true')
    # add('--bf16', action='store_true')
    args = parser.parse_args()

    partition_dataset(args.dataset, args.root_dir, args.num_partitions,
                      args.recursive)
