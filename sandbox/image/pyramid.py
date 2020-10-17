from . import Image
from scipy import ndimage  # type: ignore


class Pyramid:
    """
    A Pyramid object represents a Gaussian image pyramid.
    """

    def __init__(self, image: Image, levels: int = 3) -> None:
        """
        Constructs a Gaussian Pyramid with the given image and number of levels.

        Args:
            image: The base Image of this Pyramid.
            levels: The number of levels in this Pyramid.
        """
        modulus = 2**(levels - 1)
        assert levels > 0, 'Pyramid must have a strictly positive numbers of levels.'
        assert image.width % modulus == image.height % modulus == 0, f'Image dimensions must be divisible by {modulus}.'

        self.__levels = levels
        self.__layers = [image.copy()]
        for level in range(1, levels):
            layer = self.__layers[-1].copy()
            # Apply a Gaussian filter to each colour channel independently.
            for channel in range(3):
                layer[:, :][:, :, channel] = ndimage.gaussian_filter(layer[:, :, channel], sigma=1, truncate=2)
            layer.resize(layer.width // 2, layer.height // 2)
            self.__layers.append(layer)

    @property
    def levels(self) -> int:
        """Returns the number of levels in this Pyramid."""
        return self.__levels

    def reconstruct(self) -> Image:
        """Reconstructs the full-resolution image from this Pyramid."""
        return self.__layers[0]

    def __getitem__(self, key: object) -> Image:
        """Returns the Image at the given level in this Pyramid."""
        if isinstance(key, int):
            return self.__layers[key]
        else:
            raise NotImplementedError(f"Pyramid.__getitem__() is not implemented for type {type(key)}.")

    def __setitem__(self, key: object, image: Image) -> None:
        """Sets the Image at the given level in this Pyramid to the specified Image."""
        if isinstance(key, int):
            self.__layers[key] = image
        else:
            raise NotImplementedError(f"Pyramid.__setitem__() is not implemented for type {type(key)}.")
