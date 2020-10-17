import numpy as np  # type: ignore
import random       # type: ignore
import tqdm         # type: ignore

from . import TextureSynthesizer
from image import Image, Point
from scipy import signal  # type: ignore
from typing import List, Tuple


class SpiralPixelNeighbourhoodSynthesizer(TextureSynthesizer):
    """
    A SpiralPixelNeighbourhoodSynthesizer object synthesizes textures using the
    algorithm from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf.

    The basic idea is to seed the output Image with part of the source Image and
    then fill the remaining pixels one-by-one by finding partial neighbourhood
    matches from the output Image and the source Image.
    """

    def __init__(self,
                 source_image_path: str,
                 source_image_size: Tuple[int, int],
                 output_image_path: str,
                 output_image_size: Tuple[int, int],
                 seed_padding: int,
                 neighbourhood_padding: List[int]) -> None:
        """
        Constructs a new SpiralPixelNeighbourhoodSynthesizer object with the given
        TextureSynthesizer parameters, seed padding, and neighbourhood padding.

        Args:
            source_image_path: See TextureSynthesizer.__init__().
            source_image_size: See TextureSynthesizer.__init__().
            output_image_path: See TextureSynthesizer.__init__().
            output_image_size: See TextureSynthesizer.__init__().
            seed_padding: The half-width of the texture seed window.
            neighbourhood_padding: The half-width of a pixel neighbourhood.
        """
        super().__init__(source_image_path, source_image_size, output_image_path, output_image_size)
        self.__seed_padding = seed_padding
        self.__neighbourhood_padding = neighbourhood_padding[0]
        assert len(neighbourhood_padding) == 1, "Exactly one neighbourhood padding size must be specified."

    def render(self, source_image: Image, output_image: Image) -> None:
        """See TextureSynthesizer.render()."""
        self.__seed_output_image(source_image, output_image)
        for point in tqdm.tqdm(output_image.cover('spiral'), total=output_image.area):
            if not output_image.filled(point, 0)[0, 0]:
                self.__render_output_pixel(source_image, output_image, point)

    def __seed_output_image(self, src_image: Image, out_image: Image) -> None:
        """
        Seeds the given output Image using a random window from the provided
        source Image.

        Args:
            src_image: The source Image.
            out_image: The output Image.
        """
        # Extract a seed window from the source Image.
        _, src_max_corner = src_image.corners()
        src_seed_x = random.randint(self.__seed_padding, src_max_corner.x - self.__seed_padding - 1)
        src_seed_y = random.randint(self.__seed_padding, src_max_corner.y - self.__seed_padding - 1)
        src_seed_center = Point(src_seed_x, src_seed_y)
        src_seed_window = src_image.extract(src_seed_center, self.__seed_padding)

        # Paste the seed to the center of the output Image.
        out_seed_center = out_image.center
        out_image.paste(out_seed_center, src_seed_window)

    def __render_output_pixel(self, src_image: Image, out_image: Image, out_point: Point) -> None:
        """
        Renders the given pixel in the output Image using the colour of a pixel
        from the source Image with a similar neighbourhood to the output pixel.

        Args:
            src_image: The source Image.
            out_image: The output Image.
            out_point: The pixel to be rendered.
        """
        padding = self.__neighbourhood_padding

        # The output window serves as a reference for the neighbourhood matching.
        out_filled = out_image.filled(out_point, padding)
        out_window = out_image.extract(out_point, padding)

        # Compute a 2D Gaussian kernel for the squared pixel differences.
        gaussian_1D = signal.gaussian(2 * padding + 1, std=padding)
        gaussian_2D = np.outer(gaussian_1D, gaussian_1D)
        gaussian_2X = np.extract(out_filled, gaussian_2D)

        # Calculate the distance score for each neighbourhood in the source Image.
        distances = self._apply_distance_filter(src_image, out_window, out_filled, gaussian_2X)
        threshold = 1.1 * distances.min()

        # Set the colour of the output pixel to the colour of a random pixel
        # from the source Image with a similar neighbourhood.
        candidates = np.argwhere(distances <= threshold)
        candidate = candidates[np.random.choice(len(candidates))]
        out_image[out_point] = src_image[tuple(candidate)]
