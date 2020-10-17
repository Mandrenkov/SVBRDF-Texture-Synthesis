import abc
import image
import math
import torch
import utils

from torch import Tensor


class Light(abc.ABC):
    '''
    The Light class represents a lighting model.
    '''

    @abc.abstractmethod
    def directions(self, points: Tensor) -> Tensor:
        '''
        Returns the incident directions of this Light with respect to the given points.

        Args:
            points: Tensor [R, C, 3] of points.

        Returns:
            Tensor [R, C, D, 3] of incident directions from each point to the Light.
        '''
        raise NotImplementedError('Class "Light" does not implement method "directions".')

    @abc.abstractmethod
    def radiance(self, points: Tensor) -> Tensor:
        '''
        Returns the incident radiance of this Light with respect to the given points.

        Args:
            points: Tensor [R, C, 3] of points.

        Returns:
            Tensor [R, C, D, 3] of incident radiance on each point from the Light.
        '''
        raise NotImplementedError('Class "Light" does not implement method "radiance".')


class PunctualLight(Light):
    '''
    The PunctualLight class represents a punctual (i.e., point) light.
    '''

    def __init__(self, position: Tensor, lumens: Tensor) -> None:
        '''
        Constructs a new PunctualLight at the given position with provided flux.

        Args:
            position: Tensor [3] representing the position of the PunctualLight.
            lumens: Tensor [3] representing the luminous flux of the PunctualLight.
        '''
        assert position.shape == (3,), 'Position must be an (X, Y, Z) triplet.'
        assert lumens.shape == (3,), 'Lumens must be an (R, G, B) triplet.'
        assert torch.all(lumens >= 0), 'Lumen values cannot be negative.'
        self._position = position
        self._lumens = lumens

    @property
    def position(self) -> Tensor:
        '''Returns the position of this PunctualLight.'''
        return self._position

    @property
    def lumens(self) -> Tensor:
        '''Returns the luminous flux of this PunctualLight.'''
        return self._lumens

    def directions(self, points: Tensor) -> Tensor:
        '''See Light.directions().'''
        incident_directions = -(points - self.position)
        incident_directions /= torch.norm(incident_directions, dim=-1, keepdim=True)
        return torch.unsqueeze(incident_directions, dim=2)

    def radiance(self, points: Tensor) -> Tensor:
        '''See Light.radiance().'''
        # The incident radiance on each point is given by the inverse square law.
        squared_distances = torch.sum((points - self.position).square(), dim=-1, keepdim=True)
        expanded_radiance = self._lumens.reshape(1, 1, 3).expand(*points.shape)
        return torch.unsqueeze(expanded_radiance / squared_distances, dim=2)


class DirectionalLight(Light):
    '''
    The DirectionalLight class represents a directional light.
    '''

    def __init__(self, direction: Tensor, lumens: Tensor) -> None:
        '''
        Constructs a new DirectionalLight with the given direction and flux.

        Args:
            direction: Tensor [3] representing the direction of the DirectionalLight.
            lumens: Tensor [3] representing the luminous flux of the DirectionalLight.
        '''
        assert direction.shape == (3,), 'Direction must be an (X, Y, Z) triplet.'
        assert lumens.shape == (3,), 'Lumens must be an (R, G, B) triplet.'
        assert torch.all(lumens >= 0), 'Lumen values cannot be negative.'
        self._direction = direction / torch.norm(direction)
        self._lumens = lumens

    @property
    def direction(self) -> Tensor:
        '''Returns the direction of this DirectionalLight.'''
        return self._direction

    @property
    def lumens(self) -> Tensor:
        '''Returns the luminous flux of this DirectionalLight.'''
        return self._lumens

    def directions(self, points: Tensor) -> Tensor:
        '''See Light.directions().'''
        return -self.direction.expand(points.size(0), points.size(1), 1, 3)

    def radiance(self, points: Tensor) -> Tensor:
        '''See Light.radiance().'''
        return self.lumens.expand(points.size(0), points.size(1), 1, 3)


class ImageLight(Light):
    '''
    The ImageLight class represents an image (i.e., environment) light.
    '''

    def __init__(self, path: str, num_samples: int, intensity: float) -> None:
        '''
        Constructs a new ImageLight from the given image path, number of samples, and intensity.

        Args:
            path: Path to an environment map image.
            num_samples: Number of samples to take from the environment map.
            intensity: Scalar applied to each environment map radiance sample.
        '''
        assert path, "Path cannot be empty or set to None."
        assert num_samples > 0, "Number of samples must be greater than zero."
        self._image = image.load(path, 'sRGB')
        self._intensity = intensity
        # Construct a grid of azimuth (ϕ) and zenith (θ) angles uniformly distributed over the upper unit hemisphere.
        self._samples = torch.stack([2 * math.pi * torch.rand(num_samples, device=utils.get_device_name()),
                                     torch.acos(torch.rand(num_samples, device=utils.get_device_name()))], dim=1)

    @property
    def image(self) -> Tensor:
        '''Returns the image associated of this ImageLight.'''
        return self._image

    @property
    def intensity(self) -> float:
        '''Returns the intensity associated with this ImageLight.'''
        return self._intensity

    @property
    def samples(self) -> Tensor:
        '''Returns the (ϕ, θ) samples associated with this ImageLight.'''
        return self._samples

    def directions(self, points: Tensor) -> Tensor:
        '''See Light.directions().'''
        # Convert the sampled angles into Cartesian coordinates.
        incident_directions = torch.stack([torch.cos(self.samples[:, 0]) * torch.sin(self.samples[:, 1]),
                                           torch.sin(self.samples[:, 0]) * torch.sin(self.samples[:, 1]),
                                           torch.cos(self.samples[:, 1])], dim=1).view(-1, 3)
        return incident_directions.expand(points.size(0), points.size(1), -1, 3)

    def radiance(self, points: Tensor) -> Tensor:
        '''See Light.radiance().'''
        num_samples = self.samples.size(0)
        # The bilinear sampling function operates over [N, C, H, W] Tensors.
        sample_input = self._image.permute(2, 0, 1).unsqueeze(0)
        # Convert the sampled angles into environment map coordinates.
        sample_grid = torch.zeros(size=(1, 1, num_samples, 2), device=utils.get_device_name())
        sample_grid[0, 0, :, 0] = self.samples[:, 0] / math.pi - 1
        sample_grid[0, 0, :, 1] = self.samples[:, 1] / math.pi * 2 - 1
        # Return the incident radiance from the environment map.
        incident_radiance = torch.nn.functional.grid_sample(sample_input, sample_grid, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        return self.intensity / num_samples * incident_radiance.view(-1, 3).expand(points.size(0), points.size(1), -1, 3)
