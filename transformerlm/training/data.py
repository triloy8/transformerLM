from typing import Iterable
import numpy.typing as npt
import torch
import random


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    dataset_len = dataset.shape[0]
    sampled_sequence_stack = []
    sampled_ids_stack = []
    for _ in range(batch_size):
        starting_index = random.randint(0, dataset_len - context_length - 1)
        np_sampled_sequence = dataset[starting_index:starting_index + context_length]
        np_sampled_ids = dataset[starting_index + 1:starting_index + context_length + 1]

        sampled_sequence = torch.from_numpy(np_sampled_sequence).to(device)
        sampled_ids = torch.from_numpy(np_sampled_ids).to(device)

        sampled_sequence_stack.append(sampled_sequence)
        sampled_ids_stack.append(sampled_ids)

    batch_sampled_sequence = torch.stack(sampled_sequence_stack, dim=0)
    batch_sampled_ids = torch.stack(sampled_ids_stack, dim=0)

    return batch_sampled_sequence, batch_sampled_ids

__all__ = ["get_batch"]
