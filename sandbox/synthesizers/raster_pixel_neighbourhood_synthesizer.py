import numpy as np  # type: ignore
import tqdm         # type: ignore

from . import TextureSynthesizer
from image import Image, Point, Pyramid
from scipy import signal  # type: ignore
from search import PixelNeighbourhoodTree
from typing import List, Tuple


class RasterPixelNeighbourhoodSynthesizer(TextureSynthesizer):
    """
    A RasterPixelNeighbourhoodSynthesizer object synthesizes textures using the
    algorithm from https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf.

    The basic idea is to seed the output Image with noise from the source Image
    and then resolve the pixels one-by-one in scanline order by finding the best
    neighbourhood match from the output Image to the source Image.
    """

    def __init__(self,
                 source_image_path: str,
                 source_image_size: Tuple[int, int],
                 output_image_path: str,
                 output_image_size: Tuple[int, int],
                 neighbourhood_padding: List[int],
                 tsvq_branching_factor: int = 0) -> None:
        """
        Constructs a new RasterPixelNeighbourhoodSynthesizer object with the given
        TextureSynthesizer parameters and neighbourhood padding.

        Args:
            source_image_path: See TextureSynthesizer.__init__().
            source_image_size: See TextureSynthesizer.__init__().
            output_image_path: See TextureSynthesizer.__init__().
            output_image_size: See TextureSynthesizer.__init__().
            neighbourhood_padding: The half-width of the pixel neighbourhoods.
            tsvq_branching_factor: The branching factor of the TSVQ tree.  Setting
                                   this value to 0 disables TSVQ acceleration.
        """
        super().__init__(source_image_path, source_image_size, output_image_path, output_image_size)
        self.__levels = len(neighbourhood_padding)
        self.__neighbourhood_padding = neighbourhood_padding
        self.__tsvq_branching_factor = tsvq_branching_factor
        assert self.__levels > 0, "At least one neighbourhood padding size must be specified."
        assert self.__tsvq_branching_factor >= 0, "TSVQ branching factor cannot be negative."

    def render(self, source_image: Image, output_image: Image) -> None:
        """See TextureSynthesizer.render()."""
        self.__seed_output_image(source_image, output_image)
        if self.__tsvq_branching_factor > 0:
            for padding in self.__neighbourhood_padding:
                width = 2 * padding + 1
                neighbours = np.arange(width * width).reshape((width, width)) <= (width * width // 2)
                pixel_tree = PixelNeighbourhoodTree(source_image, self.__tsvq_branching_factor, neighbours, 'Analysis [TSVQ]')
                for out_point in tqdm.tqdm(output_image.cover('raster'), desc=f'Synthesis [Padding {padding}]', total=output_image.area):
                    src_point = pixel_tree.query(output_image, out_point, 'wrap')
                    output_image[out_point] = source_image[src_point]
        else:
            src_pyramid = Pyramid(source_image, len(self.__neighbourhood_padding))
            out_pyramid = Pyramid(output_image, len(self.__neighbourhood_padding))
            for level in reversed(range(self.__levels)):
                for point in tqdm.tqdm(out_pyramid[level].cover('raster'), desc=f'Synthesis [Level {level}]', total=out_pyramid[level].area):
                    self.__render_output_pixel(src_pyramid, out_pyramid, level, point)
            output_image[:, :] = out_pyramid.reconstruct()[:, :]

    def __seed_output_image(self, src_image: Image, out_image: Image) -> None:
        """
        Seeds the given output Image with random pixels from the source Image.

        Args:
            src_image: The source Image.
            out_image: The output Image.
        """
        src_pixel_array = src_image[:, :].reshape((src_image.area, 3))
        src_index_array = np.random.choice(np.arange(src_image.area), out_image.area)
        out_image[:, :] = np.take(src_pixel_array, src_index_array, axis=0).reshape(out_image.shape)

    def __render_output_pixel(self, src_pyramid: Pyramid, out_pyramid: Pyramid, level: int, out_point: Point) -> None:
        """
        Renders the given pixel in the specified layer of the output Pyramid
        using the colour of a pixel from the source Pyramid with the closest
        neighbourhood to the output pixel.

        Args:
            src_image: The source Pyramid.
            out_image: The output Pyramid.
            level: The level of the pixel in the output Pyramid.
            out_point: The pixel to be rendered.
        """
        if level == self.__levels - 1:
            distances = self.__make_distance_matrix(src_pyramid[level], out_pyramid[level], self.__neighbourhood_padding[level], out_point, True)
        else:
            prev_distances = self.__make_distance_matrix(src_pyramid[level + 1], out_pyramid[level + 1], self.__neighbourhood_padding[level + 1], out_point // 2, False)
            next_distances = self.__make_distance_matrix(src_pyramid[level], out_pyramid[level], self.__neighbourhood_padding[level], out_point, True)
            distances = next_distances + np.kron(prev_distances, np.ones((2, 2)))
        
        candidate = np.unravel_index(np.argmin(distances), distances.shape)
        out_pyramid[level][out_point] = src_pyramid[level][candidate]

    def __make_distance_matrix(self, src_image: Image, out_image: Image, padding: int, out_point: Point, causal: bool) -> np.ndarray:
        """
        Returns a matrix containing the weighted squared difference of the pixel
        values between each window in the source Image and the window extracted
        from the output Image at the specified Point with the given padding.

        Args:
            src_image: The source Image.
            out_image: The output Image.
            padding: The padding of a pixel neighbourhood.
            out_point: The center of the reference window in the output Image.
            causal: Only use the top "L" section of each neighbourhood.
        """
        # Extract the reference window and for the neighbourhood matching.
        out_window = out_image.extract(out_point, padding, 'wrap')
        out_filled = out_image.filled(out_point, padding, 'wrap', causal)

        # Construct a 2D Gaussian kernel that matches the padding size.
        gaussian_1D = signal.gaussian(2 * padding + 1, std=padding)
        gaussian_2D = np.outer(gaussian_1D, gaussian_1D)
        gaussian_2X = np.extract(out_filled, gaussian_2D)

        # Return the weighted squared difference of each neighbourhood in the
        # source Image with respect to the reference window.
        return self._apply_distance_filter(src_image, out_window, out_filled, gaussian_2X)
