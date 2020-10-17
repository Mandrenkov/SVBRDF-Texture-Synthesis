import torch

from torch import Tensor


def normalize(tensor: Tensor) -> Tensor:
    '''
    Normalizes the given Tensor.

    Args:
        tensor: Tensor to normalize.

    Returns:
        Normalized version of the given Tensor.
    '''
    return tensor / torch.norm(tensor, dim=-1, keepdim=True)


def dot(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    '''
    Computes the dot product between the given Tensors.

    Args:
        tensor_1: The first argument to the dot product.
        tensor_2: The second argument to the dot product.

    Returns:
        Dot product of the given Tensors, preserving the final dimension.
    '''
    return torch.sum(tensor_1 * tensor_2, dim=-1, keepdim=True)
