import image
import logging
import math
import numpy  # type: ignore
import scipy.optimize  # type: ignore
import torch
import tqdm  # type: ignore
import typing
import utils

from torch import Tensor
from typing import List, Tuple
from svbrdf import SVBRDF


def fit(input_path: str, output_path: str, optimizer: str, svbrdf: SVBRDF) -> None:
    '''
    Fits the given SVBRDF to each material in the MERL 100 dataset using the provided hyperparameters.

    Args:
        input_path: Path to the MERL 100 BRDF slice image.
        output_path: Path to the desired output image.
        optimizer: SciPy optimizer to use to fit the SVBRDF.
        svbrdf: SVBRDF model to fit.
    '''
    brdfs = _load_MERL_100_BRDF_slices(input_path)
    # It is known a priori that each BRDF slice has a width and height of 60 pixels.
    normals, incident_directions, outbound_directions = _generate_SVBRDF_directions(num_rows=60, num_cols=60)
    # There is a 10 pixel gap between each BRDF slice in a pair and a 30 pixel gap between BRDF slice pairs.
    comparison = torch.zeros(30 + 10 * 90, 30 + 10 * 160, 3)
    for i, want_brdf in tqdm.tqdm(enumerate(brdfs), desc='SVBRDF Optimization', total=100, disable=logging.root.level == logging.DEBUG):
        # Each SVBRDF parameter has a valid range of [0, 1]; however, the optimizers are bad at handling extremes.
        guess = torch.rand(svbrdf.depth).numpy() / 5 + 0.4
        bounds = scipy.optimize.Bounds(numpy.array([0.01] * svbrdf.depth), numpy.array([0.99] * svbrdf.depth))
        # Find an "optimal" set of parameters for the SVBRDF which minimizes the MSE loss with respect to the material BRDF slice.
        solution = scipy.optimize.minimize(fun=_evaluate_function, jac=_evaluate_gradient, x0=guess,
                                           args=(svbrdf, normals, incident_directions, outbound_directions, want_brdf),
                                           method=optimizer, bounds=bounds)
        logging.debug('Material %d was fit with an MSE loss of %.6f', i, solution.fun)
        # Compute the BRDF slice associated with the optimized parameters.
        svbrdf.parameters = torch.from_numpy(solution.x).expand(1, normals.size(0), normals.size(1), -1)
        have_brdf = svbrdf(normals, incident_directions, outbound_directions).squeeze().clamp(0, 1)
        # Copy the MERL 100 BRDF slice alongside the optimized BRDF slice into the output image.
        row = 30 + 90 * (i // 10)
        col = 30 + 160 * (i % 10)
        comparison[row:row + 60, col:col + 60, :] = want_brdf
        comparison[row:row + 60, col + 70:col + 130, :] = have_brdf
    image.save(path=output_path, image=comparison, encoding='sRGB')


def _load_MERL_100_BRDF_slices(path: str) -> List[Tensor]:
    '''
    Loads the BRDF slices from the MERL 100 dataset image located at the given path.

    Args:
        path: Path to the MERL 100 BRDF slice image.

    Returns:
        List of Tensors [60, 60, 3] representing the BRDF slice of each material in the MERL 100 dataset.
    '''
    atlas = image.load(path=path, encoding='sRGB')
    brdfs = []
    # The row and column coordinates were manually extracted from the BRDF slice image.
    for row in (23, 107, 190, 274, 358, 441, 525, 609):
        for col in (25, 109, 192, 276, 360, 443, 527, 610, 694, 777, 861, 945, 1029):
            brdf = atlas[row:row + 60, col:col + 60]
            brdfs.append(brdf)
            if len(brdfs) == 100:
                return brdfs
    raise Exception('Number of BRDF slices in the MERL 100 dataset is less than 100.')


def _generate_SVBRDF_directions(num_rows: int, num_cols: int) -> Tuple[Tensor, Tensor, Tensor]:
    '''
    Generates a set of normals and incident-outbound direction pairs which match the θ[h] and θ[d] parameterization of the
    MERL 100 BRDF slices.

    Args:
        num_rows: Number of θ[d] samples in the BRDF slice.
        num_cols: Number of θ[h] samples in the BRDF slice.
    
    Returns:
        Tensor [R, C, 1, 3] of normals, incident directions, and outbound directions.
    '''
    normals = torch.tensor([0, 0, 1], dtype=torch.float64).expand(num_rows, num_cols, 3)
    # An incident direction corresponding to a (θ[h], θ[d]) pair is initialized by moving from the normal along the
    # positive X-axis until its zenith angle is θ[d].  The incident direction is then rotated along the positive Y-axis
    # by θ[h] degrees.
    angles = utils.create_grid(num_rows, num_cols)[:, :, :2].double() * (math.pi / 2)
    half_angles = angles[:, :, 0]
    diff_angles = angles[:, :, 1].flip(0)
    incident_directions = torch.stack([torch.sin(diff_angles),
                                       torch.sin(half_angles) * torch.cos(diff_angles),
                                       torch.cos(half_angles) * torch.cos(diff_angles)], dim=2)
    # The outbound direction corresponding to an incident direction is found by flipping its X-coordinate.
    outbound_directions = incident_directions * torch.tensor([-1, 1, 1])
    return normals.unsqueeze(2), incident_directions.unsqueeze(2), outbound_directions.unsqueeze(2)


def _evaluate_loss(parameters, svbrdf: SVBRDF, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor, want_brdf: Tensor) -> Tensor:
    '''
    Evaluates the loss function corresponding to the given SVBRDF for the specified parameters, SVBRDF arguments, and
    the desired BRDF slice.

    Args:
        parameters: Tensor [R, C, X] (or NumPy array) containing the parameters of the SVBRDF.
        svbrdf: SVBRDF to evaluate.
        normals: Tensor [R, C, 1, 3] of surface normals to be supplied to the SVBRDF.
        incident_directions: Tensor [R, C, 1, 3] of incident directions to be supplied to the SVBRDF.
        incident_directions: Tensor [R, C, 1, 3] of outbound directions to be supplied to the SVBRDF.
        want_brdf: Tensor [60, 60, 3] representing a BRDF slice from the MERL 100 dataset.

    Returns:
        The L2 loss between the BRDF slice generated by the SVBRDF and the desired BRDF slice.
    '''
    if isinstance(parameters, numpy.ndarray):
        parameters = torch.from_numpy(parameters)
    svbrdf.parameters = parameters.expand(1, normals.size(0), normals.size(1), -1)
    have_brdf = svbrdf(normals, incident_directions, outbound_directions).squeeze()
    return torch.mean((255 * have_brdf.clamp(0, 1) - 255 * want_brdf)**2)


def _evaluate_function(parameters, svbrdf: SVBRDF, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor, want_brdf: Tensor) -> float:
    '''
    Casts the result of _evaluate_loss() into a float.

    Args:
        parameters: See _evaluate_loss().
        svbrdf: See _evaluate_loss().
        normals: See _evaluate_loss().
        incident_directions: See _evaluate_loss().
        incident_directions: See _evaluate_loss().
        want_brdf: See _evaluate_loss().

    Returns:
        The floating-point representation of the result from _evaluate_loss().
    '''
    return float(_evaluate_loss(parameters, svbrdf, normals, incident_directions, outbound_directions, want_brdf))


def _evaluate_gradient(arguments: numpy.ndarray, svbrdf: SVBRDF, normals: Tensor, incident_directions: Tensor, outbound_directions: Tensor, want_brdf: Tensor) -> Tensor:
    '''
    Computes the gradient of the loss function defined in _evaluate_loss() with respect to the given arguments.

    Args:
        parameters: NumPy array containing the parameters of the SVBRDF.
        svbrdf: See _evaluate_loss().
        normals: See _evaluate_loss().
        incident_directions: See _evaluate_loss().
        incident_directions: See _evaluate_loss().
        want_brdf: See _evaluate_loss().

    Returns:
        The gradient of _evaluate_loss() with respect to the given arguments.
    '''
    parameters = torch.from_numpy(arguments)
    parameters.requires_grad = True
    loss = _evaluate_loss(parameters, svbrdf, normals, incident_directions, outbound_directions, want_brdf)
    loss.backward()
    gradient = typing.cast(Tensor, parameters.grad).numpy()
    return gradient
