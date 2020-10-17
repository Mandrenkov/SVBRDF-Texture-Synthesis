import functools
import math
import torch

from torch import Tensor
from typing import Tuple


def get_device_name() -> str:
    '''
    Returns the name of the primary computation device.

    Returns:
        If CUDA is enabled, then "cuda" is returned.  Otherwise, "cpu" is returned.
    '''
    return 'cuda' if torch.backends.cudnn.enabled else 'cpu'  # type: ignore


def concatenate(head: Tensor, tail: Tensor, dim: int = 0) -> Tensor:
    '''
    Concatenates the given Tensors along the specified dimension.  This function is identical to torch.cat() except
    that the first Tensor can be empty in which case the second Tensor is returned.

    Args:
        head: Tensor [*] to concanate on the left.
        tail: Tensor [*] to concanate on the right.
        dim: Dimension along which the concatenation should occur.

    Returns:
        Tensor [*] representing a concatenation of the given Tensors.
    '''
    return tail if len(head) == 0 else torch.cat(tensors=[head, tail], dim=dim)


def create_grid(num_rows: int, num_cols: int) -> Tensor:
    '''
    Creates a grid of evenly-spaced points with the given dimensions which spans the [0, 0, 0] x [1, 1, 0] square.

    Args:
        num_rows: Number of rows in the grid.
        num_cols: Number of columns in the grid.
    
    Returns:
        Tensor [R, C, 3] representing the centers of the points in the grid.
    '''
    assert num_rows > 0 and num_cols > 0, 'Number of rows and columns must both exceed zero.'
    grid = torch.zeros((num_rows, num_cols, 3))
    rows = torch.linspace(0, 1, num_rows + 1)[:-1] + 1.0 / num_rows / 2
    cols = torch.linspace(0, 1, num_cols + 1)[:-1] + 1.0 / num_cols / 2
    grid[:, :, :2] = torch.stack(torch.meshgrid(rows, cols), dim=2).flip(-1)
    return grid


def create_orthonormal_basis(normal: Tensor) -> Tensor:
    '''
    Creates an orthonormal basis using the given normal vector.

    Args:
        normal: Tensor [3] to be included as a member of the orthonormal basis.
    
    Returns:
        Tensor [3, 3] composed of a normal, tangent, and binormal which form an orthonormal basis.  If possible, the
        plane formed by the normal and tangent vector will lie parallel the Z-axis.
    '''
    assert normal.shape == (3,), "Normal must be an (X, Y, Z) triplet."
    device = get_device_name()
    if normal[2] == 0:
        tangent = torch.tensor([0, 0, 1], dtype=torch.float, device=device)
    elif normal[0] == 0 and normal[1] == 0:
        tangent = torch.tensor([0, 1, 0], dtype=torch.float, device=device)
    else:
        tangent = torch.tensor([normal[0], normal[1], -(normal[0]**2 + normal[1]**2) / normal[2]])
    binormal = torch.cross(normal, tangent)
    return torch.stack([normal / torch.norm(normal), tangent / torch.norm(tangent), binormal / torch.norm(binormal)])


def create_radial_distance_field(num_rows: int, num_cols: int) -> Tensor:
    '''
    Creates a field with the given dimensions where the value at each position in the field indicates the normalized
    distance to the center of the field.

    Args:
        num_rows: Number of rows in the field.
        num_cols: Number of columns in the field.
    
    Returns:
        Tensor [R, C, 1] representing the radial distance field.
    '''
    positions = create_grid(num_rows=num_rows, num_cols=num_cols)[:, :, :2] - 0.5
    distances = torch.norm(positions, dim=2, keepdim=True)
    return distances / torch.max(distances)


def interpolate(tensors: Tensor, overlap: int) -> Tensor:
    '''
    Interpolates along the columns (i.e., dimension 2) of the given Tensors with the specified amount of overlap.

    Args:
        tensors: Tensor [B, R, C, D] to be interpolated across the "C" dimension.
        overlap: Integer denoting the size of the overlap between adjacent Tensors in the batch.

    Returns:
        Tensor [R, B × C - overlap × (B - 1), D] representing the interpolation result.
    '''
    assert len(tensors.shape) == 4, 'Tensor batch must have 4 dimensions.'
    assert overlap <= tensors.size(2), 'Overlap cannot exceed the size of dimension 2 of an individual Tensor.'

    def padding(tensor: Tensor, width: int) -> Tensor:
        '''
        Returns the padding which, when concatenated with the given Tensor, produces a Tensor with the specified width.

        Args:
            tensor: Tensor [R, C, D] to be padded.
            width: Integer denoting the width of the concatenated Tensor.

        Returns:
            Tensor [R, width - C, D] of zeros.
        '''
        return torch.zeros((tensor.size(0), width - tensor.size(1), tensor.size(2)), device=get_device_name())

    def blend(left_tensor: Tensor, right_tensor: Tensor) -> Tensor:
        '''
        Interpolates between the given left and right Tensors with respect to their relative positions.

        Args:
            left_tensor: Tensor [R, C.L, D] to be interpolated from the left.
            right_tensor: Tensor [R, C.R, D] to be interpolated from the right.

        Returns:
            Tensor [R, C.L + C.R - overlap, D] representing the blended Tensor.
        '''
        width = left_tensor.size(1) + right_tensor.size(1) - overlap
        padded_left_tensor = torch.cat([left_tensor, padding(left_tensor, width)], dim=1)
        padded_right_tensor = torch.cat([padding(right_tensor, width), right_tensor], dim=1)

        # The Tensors can be smoothly blended across the overlapping region as follows:
        #     +------------+---------------------+------------+
        #     |  α = 0.00  |  α = 0.00 ... 1.00  |  α = 1.00  |
        #     +------------+---------------------+------------+
        #                   <----- Overlap ----->
        alphas_gradient = torch.cat([torch.zeros(left_tensor.size(1) - overlap, device=get_device_name()),
                                    torch.linspace(0, 1, overlap, device=get_device_name()),
                                    torch.ones(right_tensor.size(1) - overlap, device=get_device_name())])
        alphas = alphas_gradient.expand(tensors.size(1), -1).unsqueeze(-1)
        return torch.lerp(padded_left_tensor, padded_right_tensor, alphas)
    return functools.reduce(blend, tensors)


def sample_cosine_hemisphere(origin: Tensor) -> Tensor:
    '''
    Returns a sample from the cosine-weighted unit hemisphere centred about the given origin in the positive Z direction.

    Args:
        origin: Origin of the unit hemisphere.

    Returns:
        Tensor [3] representing the coordinates of the sample.
    '''
    device = get_device_name()
    uniforms = torch.rand(2, device=device)
    return origin + torch.tensor([torch.cos(2 * math.pi * uniforms[1]) * torch.sqrt(uniforms[0]),
                                  torch.sin(2 * math.pi * uniforms[1]) * torch.sqrt(uniforms[0]),
                                  torch.sqrt(1 - uniforms[0])], device=device)


def sample_embedded_rectangle(num_outer_rows: int, num_outer_cols: int, num_inner_rows: int, num_inner_cols: int) -> Tuple[slice, slice]:
    '''
    Returns a sample with the dimensions of an "inner" rectangle that lies strictly within an "outer" rectangle.

    Args:
        num_outer_rows: Number of rows in the outer rectangle.
        num_outer_cols: Number of columns in the outer rectangle.
        num_inner_rows: Number of rows in the inner rectangle.
        num_inner_cols: Number of columns in the inner rectangle.

    Returns:
        Pair of slice representing the row and column ranges of the rectangular sample.
    '''
    device = get_device_name()
    row = torch.randint(low=0, high=num_outer_rows - num_inner_rows + 1, size=(1,), device=device)
    col = torch.randint(low=0, high=num_outer_cols - num_inner_cols + 1, size=(1,), device=device)
    return slice(row, row + num_inner_rows), slice(col, col + num_inner_cols)


def compute_Gram_matrix(feature_maps: Tensor) -> Tensor:
    '''
    Computes a Gram matrix for each of the given feature maps using the processes described in the Diversified Texture
    Synthesis with Feed-forward Networks and Perceptual Losses for Real-Time Style Transfer and Super-Resolution papers.

    Args:
        feature_maps: Tensor [B, D, R, C] of feature maps.
    
    Returns:
        Tensor [B, D, D] of (normalized) Gram matrices for each of the feature maps in the batch.
    '''
    batch_size, channels, height, width = feature_maps.shape
    # It is necessary to use reshape() instead of view() since GPU tensors may not be stored contiguously.
    norm_feature_maps = feature_maps - feature_maps.reshape(batch_size, -1).mean(1).reshape(batch_size, 1, 1, 1)
    flat_feature_maps = norm_feature_maps.reshape(batch_size, channels, width * height)
    return flat_feature_maps.matmul(flat_feature_maps.transpose(-1, -2)) / (channels * width * height)
