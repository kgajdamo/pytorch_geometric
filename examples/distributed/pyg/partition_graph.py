import argparse
import os
import os.path as osp

import torch
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.distributed import Partitioner


def partition_dataset(
    ogbn_dataset: str,
    root_dir: str,
    num_parts: int,
    recursive: bool = False,
):
    dataset_dir = osp.join('./data/dataset', f'{ogbn_dataset}')
    dataset = PygNodePropPredDataset(name=ogbn_dataset, root=dataset_dir)
    data = dataset[0]

    save_dir = osp.join(f'{root_dir}', f'{num_parts}-parts')

    partitions_dir = osp.join(save_dir, f'{ogbn_dataset}-partitions')
    partitioner = Partitioner(data, num_parts, partitions_dir, recursive)
    partitioner.generate_partition()
    split_idx = dataset.get_idx_split()

    print('-- Saving label ...')
    label_dir = osp.join(save_dir, f'{ogbn_dataset}-label')
    os.makedirs(label_dir, exist_ok=True)
    torch.save(data.y.squeeze(), osp.join(label_dir, 'label.pt'))

    print('-- Partitioning training indices ...')
    train_idx = split_idx['train']
    train_idx = train_idx.split(train_idx.size(0) // num_parts)
    train_part_dir = osp.join(save_dir, f'{ogbn_dataset}-train-partitions')
    os.makedirs(train_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(train_idx[i], osp.join(train_part_dir, f'partition{i}.pt'))

    print('-- Partitioning test indices ...')
    test_idx = split_idx['test']
    test_idx = test_idx.split(test_idx.size(0) // num_parts)
    test_part_dir = osp.join(save_dir, f'{ogbn_dataset}-test-partitions')
    os.makedirs(test_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(test_idx[i], osp.join(test_part_dir, f'partition{i}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-products')
    parser.add_argument('--root_dir', type=str, default='./data/products')
    parser.add_argument('--num_partitions', type=int, default=2)
    parser.add_argument('--recursive', action='store_true')
    args = parser.parse_args()

    partition_dataset(args.dataset, args.root_dir, args.num_partitions,
                      args.recursive)
