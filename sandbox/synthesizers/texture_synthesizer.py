import abc
import datetime
import numpy as np  # type: ignore

from image import Image
from numba import jit      # type: ignore
from scipy import ndimage  # type: ignore
from typing import Tuple


class TextureSynthesizer(abc.ABC):
    """
    A TextureSynthesizer object synthesizes output images of arbitrary size that
    resemble the texture captured in a source image.
    """

    def __init__(self,
                 source_image_path: str,
                 source_image_size: Tuple[int, int],
                 output_image_path: str,
                 output_image_size: Tuple[int, int]) -> None:
        """
        Constructs a TextureSynthesizer superclass object with the given source
        and output image paths, along with a source and output image size.

        Args:
            source_image_path: Path to load the source image.
            source_image_size: Size of the source image.
            output_image_path: Path to save the output image.
            output_image_size: Size of the output image.
        """
        assert source_image_size >= (1, 1), "Source image size cannot be zero or negative."
        assert output_image_size >= (1, 1), "Output image size cannot be zero or negative."
        self.__source_image = Image(source_image_path)
        self.__source_image_size = source_image_size
        self.__output_image = Image(output_image_path)
        self.__output_image_size = output_image_size
    
    def synthesize(self) -> None:
        """Synthesizes the output image from the source image."""
        # Load the source image.
        self.__source_image.load()
        self.__source_image.resize(*self.__source_image_size)
        # Create the output image.
        self.__output_image.create(*self.__output_image_size)

        beg = datetime.datetime.now()
        self.render(self.__source_image, self.__output_image)
        end = datetime.datetime.now()
        self.__output_image.save()
        duration = end - beg
        print(f'Finished texture synthesis in {duration}.')

    @abc.abstractmethod
    def render(self, source_image: Image, output_image: Image) -> None:
        """
        Renders an output Image from the given source Image.

        Args:
            source_image: The source Image.
            output_image: The output Image.
        """
        raise NotImplementedError("TextureSynthesizer.render() is not implemented.")

    @staticmethod
    def _apply_distance_filter(image: Image, window: Image, members: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Returns a matrix containing the weighted squared difference of the pixel
        values between each window in the given Image and the reference window.
        Pixels that fall outside the Image are reflected across the boundaries
        of the Image.

        Args:
            image: The Image.
            window: The reference window.
            members: Elements to compare between the windows.
            weights: The weighting of each pixel difference within a window.
        
        Returns:
            A matrix containing the desired weighted squared differences.
        """
        distances = np.zeros(image.size)
        for channel in range(3):
            img_channel = image[:, :][:, :, channel]
            win_channel = np.extract(members, window[:, :][:, :, channel])
            extras = (win_channel, weights)
            distances += ndimage.generic_filter(input=img_channel,
                                                output=np.float64,
                                                function=weighted_squared_distance,
                                                footprint=members,
                                                mode='mirror',
                                                extra_arguments=extras)
        return distances


@jit(nopython=True)
def weighted_squared_distance(a1: np.ndarray, a2: np.ndarray, weights: np.ndarray) -> float:
    """
    Returns the weighted squared difference between the given arrays.  The @jit
    annotation allows this function to be treated as a scipy.LowLevelCallable
    for ndimage.generic_filter.

    Args:
        a1: The first array.
        a2: The second array.
        weights: The weight of each difference (e.g., a Gaussian kernel).
    
    Returns:
        The weighted squared difference between the arrays.
    """
    return np.sum((a1 - a2)**2 * weights)
