import abc
import logging
import math
import torch
import torch.nn.functional

from torch import Tensor
from utils import create_grid, create_orthonormal_basis


class Camera(abc.ABC):
    '''
    The Camera class represents a camera.
    '''

    @abc.abstractmethod
    def render(self, surface: Tensor, radiance: Tensor) -> Tensor:
        '''
        Renders the given surface with the associated radiance values to a Tensor.

        Args:
            surface: Tensor [R, C, 3] of points belonging to a planar surface.
            radiance: Tensor [R, C, 3] of RGB radiance values corresponding to each point on the surface.

        Returns:
            Tensor [W, H, 3] representing the normalized pixel values of the surface image.
        '''
        raise NotImplementedError('Class "Camera" does not implement method "render".')


class IdentityCamera(Camera):
    '''
    The IdentityCamera class simply renders the provided radiance without any modifications or projections.
    '''

    def render(self, surface: Tensor, radiance: Tensor) -> Tensor:
        '''See Camera.render().'''
        logging.info('Rendered %dx%d image of a %dx%d surface', radiance.size(1), radiance.size(0), surface.size(1), surface.size(0))
        return radiance


class PerspectiveCamera(Camera):
    '''
    The PerspectiveCamera class represents a perspective camera which renders images using ray casting.
    '''

    def __init__(self, position: Tensor, direction: Tensor, field_of_view: Tensor, resolution: Tensor, exposure: float) -> None:
        '''
        Constructs a new PerspectiveCamera with the given position, direction, field of view, resolution, and sRGB setting.

        Args:
            position: Tensor [3] representing the position of the PerspectiveCamera.
            direction: Tensor [3] representing the (unnormalized) direction of the PerspectiveCamera.
            field_of_view: Tensor [2] representing the horizontal and vertical angles of view of the PerspectiveCamera.
            resolution: Tensor [W, H] of the camera resolution.
            exposure: Scalar applied to each linear RGB value in a rendered image.
        '''
        assert position.shape == (3,), 'Position must be an (X, Y, Z) triplet.'
        assert direction.shape == (3,), 'Direction must be an (X, Y, Z) triplet.'
        assert field_of_view.shape == (2,), 'Field of view must be a (W, H) pair.'
        assert torch.all((0 < field_of_view) & (field_of_view < 180)), 'Angles of view must fall in the open range (0, 180).'
        assert resolution.shape == (2,), 'Resolution must be an (X, Y) pair.'
        self._position = position
        self._direction = direction / torch.norm(direction)
        self._field_of_view = field_of_view
        self._resolution = resolution
        self._exposure = exposure

    @property
    def direction(self) -> Tensor:
        '''Returns the direction of this PerspectiveCamera.'''
        return self._direction

    @property
    def exposure(self) -> float:
        '''
        Returns the exposure of this Camera.
        '''
        return self._exposure

    @property
    def field_of_view(self) -> Tensor:
        '''Returns the field of view of this PerspectiveCamera.'''
        return self._field_of_view

    @property
    def position(self) -> Tensor:
        '''Returns the position of this PerspectiveCamera.'''
        return self._position

    @property
    def resolution(self) -> Tensor:
        '''
        Returns the [W, H] resolution of the images rendered by this Camera.
        '''
        return self._resolution

    def render(self, surface: Tensor, radiance: Tensor) -> Tensor:
        '''See Camera.render().'''
        # Determine the height and width of a surface texel.
        surface_row_step = (surface[-1, 0] - surface[0, 0]) / (surface.size(0) - 1)
        surface_col_step = (surface[0, -1] - surface[0, 0]) / (surface.size(1) - 1)
        # Compute the origin of the surface taking into account that each point represents the center of a texel.
        surface_origin = surface[0, 0] - surface_row_step / 2 - surface_col_step / 2
        # Derive the axes of the surface which are aligned with the row and column structure of the points.
        surface_row_axis = surface_row_step * surface.size(0)
        surface_col_axis = surface_col_step * surface.size(1)
        tiled_surface_row_axis = surface_row_axis.expand(int(self.resolution[1]), int(self.resolution[0]), 3)
        tiled_surface_col_axis = surface_col_axis.expand(int(self.resolution[1]), int(self.resolution[0]), 3)

        # Construct an image plane coincident with the Cartesian plane from the focal position (0, 0, 1).
        image_plane = create_grid(int(self.resolution[1]), int(self.resolution[0])) * 2 - 1
        image_plane[:, :, 0] *= torch.tan(math.pi * self.field_of_view[0] / 360)
        image_plane[:, :, 1] *= torch.tan(math.pi * self.field_of_view[1] / 360)
        # Derive an appropriate basis for the image plane based on the direction of the camera.
        image_plane_basis = create_orthonormal_basis(self.direction)
        # Transform the image plane into the reference frame of the camera.
        camera_ray_directions = image_plane_basis[2] * image_plane[:, :, [0]] - image_plane[:, :, [1]] * image_plane_basis[1] + image_plane_basis[0]

        # The intersection between a camera ray and the surface is given by the Möller–Trumbore algorithm.
        collision_mats = torch.stack([tiled_surface_row_axis, tiled_surface_col_axis, -camera_ray_directions], dim=2)
        collision_mats = torch.unsqueeze(collision_mats, dim=2).repeat(1, 1, 4, 1, 1).transpose(-1, -2)
        surface_to_camera_ray = self.position - surface_origin
        # The algorithm boils down to solving the following system of equations with Cramer's rule:
        #     [ surface_row_axis.x surface_col_axis.x -camera_ray_directions.x ]   [ A ]   [ surface_to_camera_ray.x ]
        #     | surface_row_axis.y surface_col_axis.y -camera_ray_directions.y | x | B | = | surface_to_camera_ray.y |
        #     [ surface_row_axis.z surface_col_axis.z -camera_ray_directions.z ]   [ C ]   [ surface_to_camera_ray.z ]
        collision_mats[:, :, 0, :, 0] = surface_to_camera_ray
        collision_mats[:, :, 1, :, 1] = surface_to_camera_ray
        collision_mats[:, :, 2, :, 2] = surface_to_camera_ray
        collision_dets = torch.det(collision_mats)
        # Solve the system of equations to find the collision scalars.
        collision_vars = collision_dets[:, :, :3] / collision_dets[:, :, [-1]].expand(-1, -1, 3)
        # Discard any collisions that occurred behind the camera.
        collision_vars[:, :, :2] *= collision_vars[:, :, [2]].sign().clamp(0, 1).expand(-1, -1, 2)

        # Sample the radiosity from the collision coordinates using bilinear filtering.
        image = self.exposure * torch.nn.functional.grid_sample(input=torch.unsqueeze(radiance.permute(2, 0, 1), dim=0),
                                                                grid=torch.unsqueeze(collision_vars[:, :, [1, 0]], dim=0) * 2 - 1,
                                                                mode='bilinear',
                                                                padding_mode='zeros',
                                                                align_corners=False).squeeze().permute(1, 2, 0)
        logging.info('Rendered %dx%d image of a %dx%d surface', image.size(1), image.size(0), surface.size(1), surface.size(0))
        return image
