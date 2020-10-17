import image
import os.path
import shader
import torch
import torch.utils.data
import utils
import vector

from transform import Transform
from light import Light
from svbrdf import SVBRDF
from torch import Tensor
from typing import Dict, List, Tuple
from viewer import Viewer


class Texture:
    '''
    The Texture class represents a texture through a category and name.
    '''

    def __init__(self, category: str, name: str) -> None:
        '''
        Constructs a new Texture with the given category and name.

        Args:
            category: Category of this Texture.
            name: Name of this Texture.
        '''
        self._category = category
        self._name = name

    @property
    def category(self) -> str:
        '''Returns the category of this Texture.'''
        return self._category

    @property
    def name(self) -> str:
        '''Returns the name of this Texture.'''
        return self._name

    def __eq__(self, other: object) -> bool:
        '''
        Reports whether the given texture is equivalent to this Texture.

        Args:
            other: Texture to check for equality.

        Returns:
            True if the given Texture is equivalent to this Texture.  Otherwise, False is returned.
        '''
        if isinstance(other, Texture):
            return (self.category, self.name) == (other.category, other.name)
        elif isinstance(other, dict):
            return (self.category, self.name) == (other['Category'], other['Name'])
        else:
            raise NotImplementedError(f'Texture.__eq__() is not implemented for type {type(other)}.')

    def __str__(self) -> str:
        '''
        Returns a fully-qualified string representation of this Texture, including both the Texture category and name.

        Returns:
            String of the form: "<Texture Category> - <Texture Name>".
        '''
        return '{category} - {name}'.format(category=self.category, name=self.name)


class Dataset(torch.utils.data.Dataset):
    '''
    The Dataset class represents an RGB-to-SVBRDF dataset which consists of a collection of materials along with a set
    of corresponding SVBRDF parameter maps.  Random crops are taken of each material to form a training (or testing) dataset.
    '''

    def __init__(self, dims: Dict, path: str, layout: Dict, textures: List[Texture], transforms: List[Transform], svbrdf: SVBRDF, lights: List[Light], viewer: Viewer) -> None:
        '''
        Constructs a new Dataset with the given dimensions, path, layout, textures, SVBRDF, Lights, and Viewer.

        Args:
            dims: Dimensions of the Dataset textures and crops.
            path: Path to the root directory of the Dataset.
            layout: Filesystem layout of each element in the Dataset.
            textures: Texture descriptions which comprise the Dataset.
            transforms: Transforms to be applied to a sample from the Dataset.
            svbrdf: SVBRDF associated with the parameter maps in the Dataset.
            lights: Lights used to shade a texture.
            viewer: Viewer used to shade a texture.
        '''
        for key in ('Texture', 'Crop'):
            assert key in dims, f'Dimensions dictionary is missing key "{key}".'
        for key in ('Normals', 'Parameters'):
            assert key in layout, f'Layout dictionary is missing key "{key}".'
        for i, parameter in enumerate(layout['Parameters']):
            assert 'Type' in parameter, f'Parameter {i} is missing key "Type" in layout dictionary.'
            assert 'Name' in parameter, f'Parameter {i} is missing key "Name" in layout dictionary.'
        self._dims = dims
        self._path = path
        self._layout = layout
        self._textures = textures
        self._transforms = transforms
        self._svbrdf = svbrdf
        self._lights = lights
        self._viewer = viewer
        # A rendering surface is needed to generate flash-lit images for consumption by an SVBRDF autoencoder network.
        # Similarly, a radial distance field (indicating the distance from each point on the surface to the center of
        # the surface) enables the network to discriminate between flash-lit and non-flash-lit regions.
        num_crop_rows = self._dims['Crop'][0]
        num_crop_cols = self._dims['Crop'][1]
        self._surface = utils.create_grid(num_rows=num_crop_rows, num_cols=num_crop_cols)
        self._radial_distance_field = utils.create_radial_distance_field(num_rows=num_crop_rows, num_cols=num_crop_cols).unsqueeze(0)

    @property
    def textures(self) -> List[Texture]:
        '''
        Returns the textures in this Dataset.

        Returns:
            List of textures in this Dataset.
        '''
        return self._textures

    def __len__(self) -> int:
        '''
        Returns the number of textures in this Dataset.

        Returns:
            Number of textures in this Dataset.
        '''
        return len(self.textures)

    def sample(self, material: int, quantity: int = 1) -> Tuple[Tensor, Tuple[Tensor, SVBRDF]]:
        '''
        Returns a Dataset tuple derived from the given texture with the specified batch size.

        Args:
            material: Index of the texture to sample from this Dataset.
            quantity: Number of samples to include in the returned batch.

        Returns:
            Tensor [B, 3, R, C] of cropped front-parallel "renderings" of the indicated texture which are suitable for
            consumption by an SVBRDF autoencoder, Tensor [B, R, C, 3] representing the ground-truth normal maps of the
            texture samples, and an SVBRDF embedded with the ground-truth parameter values for the texture samples.
        '''
        assert 0 <= material < len(self.textures), f'Material {material} falls outside the half-open range [0, {len(self.textures)}).'
        # Load the normals and SVBRDF parameters of the texture only once; hitting the disk is expensive!
        texture = self.textures[material]
        texture_normals = self._load_normals(texture)
        texture_parameters = self._load_parameters(texture)
        # The texture and crop dimensions are, of course, the same for each sample.
        num_texture_rows = self._dims['Texture'][0]
        num_texture_cols = self._dims['Texture'][1]
        num_crop_rows = self._dims['Crop'][0]
        num_crop_cols = self._dims['Crop'][1]
        # Initializing the batch Tensors with an empty Tensor() invokes special behaviour in utils.concatenate().
        batch_inputs = Tensor()
        batch_normals = Tensor()
        batch_parameters = Tensor()
        for sample in range(quantity):
            lights, viewer = self._lights, self._viewer
            # Wrapping or reflection crops are not supported so an embedded rectangle will suffice.
            row_crop, col_crop = utils.sample_embedded_rectangle(num_outer_rows=num_texture_rows, num_inner_rows=num_crop_rows,
                                                                 num_outer_cols=num_texture_cols, num_inner_cols=num_crop_cols)
            # The SVBRDF is cloned here to avoid polluting the global SVBRDF with arbitrary parameter values.
            svbrdf = self._svbrdf.clone()
            svbrdf.parameters = texture_parameters[:, row_crop, col_crop]
            normals = texture_normals[:, row_crop, col_crop]
            # It should not matter if each picture in the batch was produced from a different rendering context.
            for transform in self._transforms:
                normals, svbrdf.parameters, lights, viewer = transform.apply(normals, svbrdf.parameters, lights, viewer)
            # The surface radiance can be interpreted as a perspective rendering from the location of the Viewer.
            radiance = shader.shade(surface=self._surface, normals=normals, lights=lights, viewer=viewer, svbrdf=svbrdf)
            inputs = torch.cat([radiance, self._radial_distance_field], dim=3).permute(0, 3, 1, 2)
            # Unlike torch.cat(), utils.concatenate() gracefully handles empty tensors in the first argument.
            batch_normals = utils.concatenate(batch_normals, normals)
            batch_parameters = utils.concatenate(batch_parameters, svbrdf.parameters)
            batch_inputs = utils.concatenate(batch_inputs, inputs)
        # Clone the SVBRDF (again) to avoid overwriting parameter values in a multiprocessing context.
        batch_svbrdf = self._svbrdf.clone()
        batch_svbrdf.parameters = batch_parameters
        # The structure of the Dataset tuple suggests an association between the normals and SVBRDF.
        return batch_inputs, (batch_normals, batch_svbrdf)

    def __getitem__(self, material: int) -> Tuple[Tensor, Tuple[Tensor, SVBRDF]]:
        '''
        Returns a Dataset tuple derived from the given texture.

        Args:
            material: Index of the texture to get from this Dataset.

        Returns:
            Output of self.sample() with the removal of the batch dimensions in the pictures and normal maps.
        '''
        batch_pictures, (batch_normals, batch_svbrdf) = self.sample(material)
        return batch_pictures[0], (batch_normals[0], batch_svbrdf)

    def _derive_path_to_texture_directory(self, texture: Texture) -> str:
        '''
        Derives the path to the directory containing the contents of the given texture.

        Args:
            texture: Texture of interest.

        Returns:
            Path to the texture directory.
        '''
        return os.path.join(self._path, '{category} - {name}'.format(category=texture.category, name=texture.name))

    def _load_normals(self, texture: Texture) -> Tensor:
        '''
        Loads the normal map associated with the given texture.

        Args:
            texture: Texture of interest.

        Returns:
            Tensor [1, R, C, 3] of texture normals.
        '''
        path_to_texture = self._derive_path_to_texture_directory(texture)
        path_to_normals = os.path.join(path_to_texture, self._layout['Normals'])
        return vector.normalize(2 * image.load(path_to_normals).unsqueeze(0) - 1)

    def _load_parameters(self, texture: Texture) -> Tensor:
        '''
        Loads the parameter maps associated with the given texture.

        Args:
            texture: Texture of interest.

        Returns:
            Tensor [R, C, D] of parameter maps.
        '''
        path_to_texture = self._derive_path_to_texture_directory(texture)
        parameters = Tensor()
        for (i, parameter_config) in enumerate(self._layout['Parameters']):
            path_to_parameter = os.path.join(path_to_texture, parameter_config['Name'])
            parameter = image.load(path_to_parameter, parameter_config['Type']).unsqueeze(0)
            parameters = parameter if parameters.size(0) == 0 else torch.cat([parameters, parameter], dim=3)
        return parameters
