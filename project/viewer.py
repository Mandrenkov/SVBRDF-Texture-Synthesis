import abc
import torch

from torch import Tensor


class Viewer(abc.ABC):
    '''
    The Viewer class represents an viewing model.
    '''

    @abc.abstractmethod
    def directions(self, points: Tensor) -> Tensor:
        '''
        Returns the outbound directions of this Viewer with respect to the given points.

        Args:
            points: Tensor [R, C, 3] of points.

        Returns:
            Tensor [R, C, 3] of outbound directions from each point to the Viewer.
        '''
        raise NotImplementedError('Class "Viewer" does not implement method "directions".')


class OrthographicViewer(Viewer):
    '''
    The OrthographicViewer class represents a Viewer that perceives points under an orthographic projection.
    '''

    def __init__(self, direction: Tensor) -> None:
        '''
        Constructs a new OrthographicViewer in the given direction.

        Args:
            direction: Tensor [R, C, 3] in the direction the OrthographicViewer is facing.
        '''
        assert direction.shape == (3,), 'Direction must be an (x, y, z) triplet.'
        self._direction = direction / torch.norm(direction)

    @property
    def direction(self) -> Tensor:
        '''Returns the direction of this OrthographicViewer.'''
        return self._direction

    def directions(self, points: Tensor) -> Tensor:
        '''See Viewer.directions().'''
        return -torch.ones(points.shape) * self.direction


class PerspectiveViewer(Viewer):
    '''
    The PerspectiveViewer class represents a Viewer that perceives points under an perspective projection.
    '''

    def __init__(self, position: Tensor) -> None:
        '''
        Constructs a new PerspectiveViewer at the given position.

        Args:
            position: Tensor [3] representing the position of the PerspectiveViewer.
        '''
        assert position.shape == (3,), 'Position must be an (X, Y, Z) triplet.'
        self._position = position

    @property
    def position(self) -> Tensor:
        '''Returns the position of this PerspectiveViewer.'''
        return self._position

    def directions(self, points: Tensor) -> Tensor:
        '''See Viewer.directions().'''
        outbound_directions = -(points - self.position)
        outbound_directions /= torch.norm(outbound_directions, dim=-1, keepdim=True)
        return outbound_directions

