import imageio  # type: ignore
import logging
import numpy  # type: ignore
import os
import pathlib
import torch
import torchvision  # type: ignore
import utils

from torch import Tensor
from typing import Callable


def load(path: str, encoding: str = 'RGB') -> Tensor:
    '''
    Loads the image at the given path using the supplied encoding.

    Args:
        path: Path to the image.
        encoding: Encoding of the image.

    Returns:
        Tensor [R, C, X] representing the normalized pixel values in the image.
    '''
    assert path, "Path cannot be empty or set to None."
    array = imageio.imread(path)
    device = utils.get_device_name()
    image = torchvision.transforms.ToTensor()(array).to(device).permute(1, 2, 0)[:, :, :3]
    if encoding == 'sRGB':
        image = convert_sRGB_to_RGB(image)
    elif encoding == 'Greyscale':
        image = convert_RGB_to_greyscale(image)
    elif encoding != 'RGB':
        raise Exception(f'Image encoding "{encoding}" is not supported."')
    logging.debug('Loaded image from "%s"', path)
    return image


def save(path: str, image: Tensor, encoding: str = 'RGB') -> None:
    '''
    Saves the given image to the specified path using the supplied encoding.

    Args:
        path: Path to the image.
        image: Tensor [R, C, X] of normalized pixel values in the image.
        encoding: Encoding of the image.
    '''
    assert path, "Path cannot be empty or set to None."
    assert torch.all(0 <= image) and torch.all(image <= 1), "Image values must fall in the closed range [0, 1]."
    if encoding == 'sRGB':
        image = convert_RGB_to_sRGB(image)
    elif encoding == 'Greyscale':
        image = convert_greyscale_to_RGB(image)
    elif encoding != 'RGB':
        raise Exception(f'Image encoding "{encoding}" is not supported."')
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, torch.floor(255 * image).detach().cpu().numpy().astype(numpy.uint8))
    logging.debug('Saved image to "%s"', path)


def clamp(function: Callable[[Tensor], Tensor]) -> Callable:
    '''
    Decorator which clamps an image destined for the given function to the range [ϵ, 1].  Note that ϵ is used in favour
    of 0 to enable differentiation through fractional exponents.

    Args:
        function: Function that accepts an image Tensor as input.

    Returns:
        Wrapper which implements the aforementioned behaviour.
    '''
    return lambda image: function(image.clamp(1E-8, 1))


@clamp
def convert_sRGB_to_RGB(image: Tensor) -> Tensor:
    '''
    Converts an sRGB image into a linear RGB image.

    Args:
        image: Tensor [R, C, 3] of an sRGB image.

    Returns:
        Tensor [R, C, 3] of a linear RGB image.
    '''
    assert len(image.shape) >= 3 and image.size(-1) == 3, 'sRGB image must have dimensionality [*, R, C, 3].'
    below = (image <= 0.04045) * image / 12.92
    above = (image > 0.04045) * ((image + 0.055) / 1.055)**2.4
    return below + above


@clamp
def convert_RGB_to_sRGB(image: Tensor) -> Tensor:
    '''
    Converts a linear RGB image into an sRGB image.

    Args:
        image: Tensor [R, C, 3] of a linear RGB image.

    Returns:
        Tensor [R, C, 3] of an sRGB image.
    '''
    assert len(image.shape) >= 3 and image.size(-1) == 3, 'RGB image must have dimensionality [*, R, C, 3].'
    below = (image <= 0.0031308) * image * 12.92
    above = (image > 0.0031308) * (1.055 * image**(1 / 2.4) - 0.055)
    return below + above


def convert_RGB_to_greyscale(image: Tensor) -> Tensor:
    '''
    Converts a linear RGB image into a greyscale image.

    Args:
        image: Tensor [R, C, 3] of an RGB image.

    Returns:
        Tensor [R, C, 1] of a greyscale image.
    '''
    assert len(image.shape) == 3 and (image.size(2) == 1 or image.size(2) == 3), 'RGB image must have dimensionality [R, C, 1] or [R, C, 3].'
    if image.size(2) == 3:
        assert torch.all((image[:, :, 0] == image[:, :, 1]) & (image[:, :, 0] == image[:, :, 2])), 'RGB image must have the same value across each colour channel.'
        return image[:, :, [0]]
    return image


def convert_greyscale_to_RGB(image: Tensor) -> Tensor:
    '''
    Converts a greyscale image into a linear RGB image.

    Args:
        image: Tensor [R, C, 1] of a greyscale image.

    Returns:
        Tensor [R, C, 3] of a linear RGB image.
    '''
    assert len(image.shape) == 3 and image.size(2) == 1, 'Greyscale image must have dimensionality [R, C, 1].'
    return image.expand(-1, -1, 3)
