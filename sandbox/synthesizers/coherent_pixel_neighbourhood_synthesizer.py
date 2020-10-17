import numpy as np  # type: ignore
import tqdm         # type: ignore

from . import TextureSynthesizer
from image import Image, Point
from typing import Tuple


class CoherentPixelNeighbourhoodSynthesizer(TextureSynthesizer):
    """
    A CoherentPixelNeighbourhoodSynthesizer object synthesizes textures using the
    algorithm from https://www.cs.utah.edu/~michael/ts/ts.pdf.

    The basic idea is to first build a similarity set for each neighbourhood in
    the source Image.  Then, the output Image is seeded with noise from the source
    Image and the pixels are resolved one-by-one in scanline order by selecting
    a neighbour (in the source Image) of the neighbour in the output Image that
    most closely matches the current neighbourhood in the output Image.
    """

    def __init__(self,
                 source_image_path: str,
                 source_image_size: Tuple[int, int],
                 output_image_path: str,
                 output_image_size: Tuple[int, int],
                 coherent_set_size: int) -> None:
        """
        Constructs a new CoherentPixelNeighbourhoodSynthesizer object with the
        given TextureSynthesizer parameters and similarity set size.

        Args:
            source_image_path: See TextureSynthesizer.__init__().
            source_image_size: See TextureSynthesizer.__init__().
            output_image_path: See TextureSynthesizer.__init__().
            output_image_size: See TextureSynthesizer.__init__().
            coherent_set_size: The number of similar neighbourhoods to associate
                               with each neighbourhood in the source Image.
        """
        super().__init__(source_image_path, source_image_size, output_image_path, output_image_size)
        self.__coherent_set_size = coherent_set_size

    def render(self, source_image: Image, output_image: Image) -> None:
        """See TextureSynthesizer.render()."""
        # Initialize the output Image with random noise.
        output_source_map = self.__seed_output_image(source_image, output_image)
        neighbourhood_map = np.zeros((source_image.height, source_image.width, self.__coherent_set_size, 2), dtype=np.int64)
        for point in tqdm.tqdm(source_image.cover('raster'), total=source_image.area):
            neighbourhood_map[point.y, point.x, :] = self.__find_nearest_neighbours(source_image, point)
        for point in tqdm.tqdm(output_image.cover('raster'), total=output_image.area):
            self.__render_output_pixel(source_image, output_image, output_source_map, neighbourhood_map, point)

    def __seed_output_image(self, src_image: Image, out_image: Image) -> np.ndarray:
        """
        Seeds the given output Image with random pixels from the source Image.

        Args:
            src_image: The source Image.
            out_image: The output Image.

        Returns:
            Array mapping each pixel in the output Image to its corresponding
            pixel in the source Image.
        """
        # Randomly map Points in the output Image to Points in the source Image.
        out_src_map_x = np.random.choice(np.arange(src_image.width), size=src_image.size)
        out_src_map_y = np.random.choice(np.arange(src_image.height), size=src_image.size)
        out_src_map = np.stack((out_src_map_y, out_src_map_x), axis=2)
        # Colour the pixles in the output Image using the source Image map.
        src_pixel_array = src_image[:, :].reshape((src_image.area, 3))
        src_index_array = np.ravel_multi_index(np.transpose(np.reshape(out_src_map, (out_image.area, 2))), dims=src_image.size)
        out_image[:, :] = np.take(src_pixel_array, src_index_array, axis=0).reshape(out_image.shape)
        return out_src_map

    def __find_nearest_neighbours(self, image: Image, point: Point) -> np.ndarray:
        """
        Finds the |self.__coherent_set_size| nearest neighbours to the given
        Point in the provided Image using a neighbourhood similarity metric.

        Args:
            image: The Image.
            point: The Point.

        Returns:
            Array with the locations of the nearest neighbours to the given Point.
        """
        distances = self._apply_distance_filter(image=image,
                                                window=image.extract(point, 1, 'reflect'),
                                                members=np.full((3, 3), True),
                                                weights=np.ones(9))
        neighbours = np.argsort(distances.reshape(-1))[:self.__coherent_set_size]
        return np.stack(np.unravel_index(neighbours, image.size), axis=1)

    def __render_output_pixel(self, src_image: Image, out_image: Image, out_src_map: np.ndarray, neighbourhood_map: np.ndarray, out_point: Point) -> None:
        """
        Renders the given pixel in the output Image using the colour of a pixel
        from the source Image with a similar neighbourhood to the output pixel.

        Args:
            src_image: The source Image.
            out_image: The output Image.
            out_point: The pixel to be rendered.
        """
        src_neighbour_points = set()
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0)):
            # Derive the location of the neighbour in the output Image.
            out_neighbour = out_image.project(out_point + Point(dx, dy), 'wrap')
            # Find the location of the source Point corresponding to the neighbour.
            src_point = Point(*out_src_map[out_neighbour.y, out_neighbour.x, ::-1])
            # Add the nearest neighbours of the source point to the set of candidates.
            for src_neighbour in neighbourhood_map[src_point.y, src_point.x, :, :]:
                src_neighbour_point = src_image.project(Point(src_neighbour[1] - dx, src_neighbour[0] - dy), 'reflect')
                src_neighbour_points.add(src_neighbour_point)

        # Transform the set into a list to enable indexing.
        candidates = list(src_neighbour_points)

        # Calculate the distance score for each candidate neighbour.
        distances = np.zeros(len(candidates))
        for i, src_point in enumerate(candidates):
            distances[i] = self._apply_distance_filter(image=src_image.extract(src_point, 1, 'reflect'),
                                                       window=out_image.extract(out_point, 1, 'reflect'),
                                                       members=np.arange(9).reshape(3, 3) < 4,
                                                       weights=np.ones(4))[1, 1]

        # Set the colour of the output pixel to the colour of the source pixel
        # with the closest neighbourhood to the output pixel.  Note that the
        # output source map should also be updated to reflect this change.
        candidate = candidates[np.argmin(distances)]
        out_image[out_point] = src_image[candidate]
        out_src_map[out_point.y, out_point.x, :] = np.array([candidate.y, candidate.x])

