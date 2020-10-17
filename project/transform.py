import torch
import utils

from abc import ABC, abstractmethod
from light import Light, PunctualLight
from torch import Tensor
from typing import Dict, List, Tuple
from viewer import Viewer, PerspectiveViewer


class Transform(ABC):
    '''
    The Transform class represents a data augmentation procedure.
    '''

    @abstractmethod
    def apply(self, normals: Tensor, parameters: Tensor, lights: List[Light], viewer: Viewer) -> Tuple[Tensor, Tensor, List[Light], Viewer]:
        '''
        Applies this Transform to the given normal map, SVBRDF parameters, Lights, and Viewer.

        Args:
            normals: Tensor [1, R, C, 3] of normals.
            parameters: Tensor [B, R, C, D] of SVBRDF parameters.
            lights: Lights used to evaluate an SVBRDF.
            viewer: Viewer used to evaluate an SVBRDF.

        Returns:
            Tuple representing the transformed versions of the given arguments.
        '''
        raise NotImplementedError('Class "Transform" does not implement the "apply" method.')


class ReflectionTransform(Transform):
    '''
    The ReflectionTransform class reflects a texture about its horizontal or vertical (or both) axes.
    '''

    def apply(self, normals: Tensor, parameters: Tensor, lights: List[Light], viewer: Viewer) -> Tuple[Tensor, Tensor, List[Light], Viewer]:
        '''See Transform.apply().'''
        for dim in range(1, 3):
            if torch.rand(1) < 0.5:
                normals, parameters = normals.flip([dim]), parameters.flip([dim])
        return normals, parameters, lights, viewer


class RotationTransform(Transform):
    '''
    The RotationTransform class rotates a texture by a random multiple of 90 degrees.
    '''

    def apply(self, normals: Tensor, parameters: Tensor, lights: List[Light], viewer: Viewer) -> Tuple[Tensor, Tensor, List[Light], Viewer]:
        '''See Transform.apply().'''
        turns = int(torch.randint(low=0, high=4, size=(1,)))
        rotated_normals = normals.rot90(turns, dims=[1, 2])
        rotated_parameters = parameters.rot90(turns, dims=[1, 2])
        return rotated_normals, rotated_parameters, lights, viewer


class ElevationTransform(Transform):
    '''
    The ElevationTransform class scales the elevation of applicable Lights and Viewers.
    '''

    def __init__(self, min_scalar: float, max_scalar: float) -> None:
        '''
        Constructs a new ElevationTransform with the given minimum and maximum scalar values.

        Args:
            min_scalar: Minimum elevation scaling factor.
            max_scalar: Maximum elevation scaling factor.
        '''
        self._min_scalar = min_scalar
        self._max_scalar = max_scalar

    def apply(self, normals: Tensor, parameters: Tensor, lights: List[Light], viewer: Viewer) -> Tuple[Tensor, Tensor, List[Light], Viewer]:
        '''See Transform.apply().'''
        device = utils.get_device_name()
        scalar = torch.rand(1) * (self._max_scalar - self._min_scalar) + self._min_scalar
        # Take care not to modify the original list of Lights.
        elevated_lights = [light for light in lights]
        for i, light in enumerate(lights):
            # Only punctual Lights can be raised or lowered.
            if isinstance(light, PunctualLight):
                elevated_position = light.position * torch.tensor([1.0, 1.0, scalar], device=device)
                elevated_lights[i] = PunctualLight(position=elevated_position, lumens=light.lumens)
        elevated_viewer = viewer
        # Similarly, only perspective Viewers can be raised or lowered.
        if isinstance(viewer, PerspectiveViewer):
            elevated_position = viewer.position * torch.tensor([1.0, 1.0, scalar], device=device)
            elevated_viewer = PerspectiveViewer(position=elevated_position)
        return normals, parameters, elevated_lights, elevated_viewer


class SubstitutionTransform(Transform):
    '''
    The SubstitutionTransform class replaces a set of SVBRDF parameter maps with random fields.
    '''

    def __init__(self, targets: List[Dict]) -> None:
        '''
        Constructs a new SubstitutionTransform with the given list of SVBRDF parameter map targets.

        Args:
            targets: SVBRDF parameter map targets.
        '''
        for i, target in enumerate(targets):
            for key in ('Index', 'Min Value', 'Max Value'):
                assert key in target, f'Target {i} is missing key "{key}".'
        self._targets = targets

    def apply(self, normals: Tensor, parameters: Tensor, lights: List[Light], viewer: Viewer) -> Tuple[Tensor, Tensor, List[Light], Viewer]:
        '''See Transform.apply().'''
        replaced_parameters = parameters.clone().detach()
        for i, target in enumerate(self._targets):
            index = target['Index']
            min_value = target['Min Value']
            max_value = target['Max Value']
            uniform_random_field = torch.rand_like(parameters[:, :, :, index], device=utils.get_device_name())
            replaced_parameters[:, :, :, index] = uniform_random_field * (max_value - min_value) + min_value
        return normals, replaced_parameters, lights, viewer
