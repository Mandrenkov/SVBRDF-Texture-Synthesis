import logging
import math
import torch
import torch.nn
import torch.optim
import torchvision  # type: ignore
import vector
import utils

from torch import Tensor
from typing import Any, Callable, Dict, List, Tuple


class SVBRDFAutoencoder(torch.nn.Module):
    '''
    The SVBRDFAutoencoder class represents an autoencoder neural network which accepts a batch of RGB images as input
    and produces a set of SVBRDF parameter maps as output.
    '''

    def __init__(self, dims: Dict[str, Dict[str, List[int]]], encoders: Dict[str, torch.nn.Module], decoder: torch.nn.Module, path: str) -> None:
        '''
        Constructs a new SVBRDFAutoencoder with the given encoders, decoder, and parameter file path.

        Args:
            dims: Dictionary of input, output, and latent field dimensions.
            encoders: Dictionary of encoder neural networks with entries for the "Local", "Global", and "Periodic" keys.
                      A TypedDict is not used because there is no clean solution that supports both Python 3.7 and 3.8.
            decoder: Decoder neural network.
            path: Path to the parameters file.
        '''
        super().__init__()
        for key in ('Texture', 'Latent'):
            assert key in dims, f'Dimensions dictionary is missing key "{key}".'
        for key in ('Local', 'Global', 'Periodic'):
            assert key in dims['Latent'], f'Dimensions dictionary is missing key "{key}" under scope "Latent".'
            assert key in encoders, f'Encoders dictionary is missing key "{key}".'
        self._dims = dims
        self._path = path
        self.encoders = encoders
        self.decoder = decoder
        for key, encoder in encoders.items():
            self.add_module(f'encoders[{key}]', encoder)

    @property
    def dimensions(self) -> Dict[str, Dict[str, List[int]]]:
        '''
        Returns the expected input, output, and latent dimensions of this SVBRDFAutoencoder.

        Returns:
            Dictionary of SVBRDFAutoencoder dimensions.
        '''
        return self._dims

    def load(self) -> None:
        '''
        Loads the parameters of this SVBRDFAutoencoder.
        '''
        device = utils.get_device_name()
        self.load_state_dict(torch.load(self._path, map_location=torch.device(device)))
        logging.debug('Loaded network parameters from "%s"', self._path)

    def save(self) -> None:
        '''
        Saves the parameters of this SVBRDFAutoencoder.
        '''
        torch.save(self.state_dict(), self._path)
        logging.debug('Saved network parameters to "%s"', self._path)

    def decode(self, latents: Tensor) -> Tensor:
        '''
        Decodes the given latent batch into an SVBRDF parameter map tensor.

        Args:
            latents: Tensor [B, D, R, C] of latent fields.

        Returns:
            Tensor [B, D, R, C] of SVBRDF parameter maps.
        '''
        return self.decoder(latents)

    def encode(self, images: Tensor) -> Tensor:
        '''
        Encodes the given batch of RGB images into an set of latent fields.

        Args:
            images: Tensor [B, 3, R, C] of images.

        Returns:
            Tensor [B, D, R, C] of latent fields.
        '''
        # The "local" part of the latent is simply the output of the local encoder.
        local_field = self.encoders['Local'](images)
        height, width = local_field.size(2), local_field.size(3)
        # The "global" part of the latent must be expanded to match the height and width of the local latent.
        global_vector = self.encoders['Global'](images)
        global_field = global_vector.expand(height, width, -1, -1).permute(2, 3, 0, 1)
        # Each consecutive triplet of the periodic encoder output represents the parameters of a planar wave.
        wave_vector = self.encoders['Periodic'](global_vector)
        wave_matrix = wave_vector.view(wave_vector.size(0), wave_vector.size(1) // 3, 3)
        wave_field = wave_matrix.expand(height, width, -1, -1, -1).permute(2, 3, 0, 1, 4)
        # The planar wave field needs to be applied to the indices of each spatial position in the latent tensor.
        dtype = images.dtype
        index_mesh = torch.stack(torch.meshgrid(torch.arange(height, dtype=dtype), torch.arange(width, dtype=dtype)), dim=2)
        index_grid = torch.cat([index_mesh, torch.ones((height, width, 1), dtype=dtype)], dim=2)
        index_field = index_grid.expand(wave_field.size(0), wave_field.size(1), -1, -1, -1)
        # The "periodic" part of the latent can now be computed according to the description in the PSGAN paper.
        periodic_field = torch.sin(vector.dot(wave_field, index_field).squeeze(dim=4))
        return torch.cat([local_field, global_field, periodic_field], dim=1)

    def forward(self, images: Tensor) -> Tensor:
        '''
        Evaluates this SVBRDFAutoencoder over the given batch of RGB images to produce a set of SVBRDF parameter maps.

        Args:
            images: Tensor [B, 3, R, C] of images.

        Returns:
            Tensor [B, D, R, C] of SVBRDF parameter maps.
        '''
        latents = self.encode(images)
        return self.decode(latents)

    def derive_periodic_field(self, global_field: Tensor, first_row_index: int = 0, first_col_index: int = 0) -> Tensor:
        '''
        Derives the periodic field corresponding to the given global field.

        Args:
            global_field: Tensor [B, R, C, D.G] of global field vectors.
            first_row_index: Integer denoting the first row coordinate in the planar wave evaluation.
            first_col_index: Integer denoting the first column coordinate in the planar wave evaluation.

        Returns:
            Tensor [B, R, C, D.P] representing the derived periodic field.
        '''
        assert len(global_field.shape) == 4, 'Global field must have 4 dimensions.'
        # The batch size, height, and width of the global field are shared by the periodic field.
        batch_size, height, width = global_field.size(0), global_field.size(2), global_field.size(3)
        global_vectors = global_field.reshape(batch_size * height * width, -1)

        # Each consecutive triplet of periodic encoder output represents the parameters of a planar wave.
        wave_vectors = self.encoders['Periodic'](global_vectors)
        wave_count = self.dimensions['Latent']['Periodic'][2]
        wave_field = wave_vectors.reshape(batch_size, height, width, wave_count, 3)

        # The values in the index mesh are the locations at which the planar waves will be evaluated.
        dtype = global_field.dtype
        index_rows = torch.arange(first_row_index, first_row_index + height, dtype=dtype)
        index_cols = torch.arange(first_col_index, first_col_index + width, dtype=dtype)
        index_mesh = torch.stack(torch.meshgrid(index_rows, index_cols), dim=2)
        # Now, the shape of the index mesh must be massaged to match the wave field.
        index_grid = torch.cat([index_mesh, torch.ones((height, width, 1), dtype=dtype)], dim=2)
        index_field = index_grid.expand(batch_size, wave_count, -1, -1, -1).permute(0, 2, 3, 1, 4)

        # Finally, the periodic field can be computed according to the description in the PSGAN paper.
        return torch.sin(vector.dot(wave_field, index_field).squeeze(dim=4)).permute(0, 3, 1, 2)

    @staticmethod
    def interpret(output: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Extracts the normals and SVBRDF parameters from the output of an SVBRDFAutoencoder.

        Args:
            output: Tensor [B, D, R, C] output of an SVBRDFAutoencoder.

        Returns:
            Tensor [B, R, C, 3] of surface normals and Tensor [B, R, C, D - 2] of SVBRDF parameters.
        '''
        # Change the Tensor dimensions from [B, D, R, C] to the more familiar [B, R, C, D].
        predictions = output.permute(0, 2, 3, 1)
        # The output angles are parameterized as (θ, ϕ) spherical coordinates.
        angles = predictions[:, :, :, :2] * torch.tensor([math.pi / 2, 2 * math.pi])
        normals = torch.stack([torch.sin(angles[:, :, :, 0]) * torch.cos(angles[:, :, :, 1]),
                               torch.sin(angles[:, :, :, 0]) * torch.sin(angles[:, :, :, 1]),
                               torch.cos(angles[:, :, :, 0])], dim=3)
        parameters = predictions[:, :, :, 2:]
        return normals, parameters


class VGG19(torch.nn.Module):
    '''
    The VGG19 class represents a prefix of the traditional VGG-19 network which only extends to the depth of the deepest
    "named" layer for style or content loss purposes.  In practice, the VGG19.extract_feature_maps() method is the only
    method on a VGG19 object that needs to be invoked to compute the desired losses.
    '''

    def __init__(self) -> None:
        '''
        Constructs a new VGG19 network.
        '''
        super().__init__()
        self.layer_name_to_index_map = {
            'conv1_1': 0,
            'relu1_1': 1,
            'conv2_1': 5,
            'relu2_1': 6,
            'conv3_1': 10,
            'relu3_1': 11,
            'conv4_1': 19,
            'relu4_1': 20,
            'conv4_2': 21,
            'relu4_2': 22,
            'conv5_1': 28,
            'relu5_1': 29
        }
        deepest_layer_depth = max(self.layer_name_to_index_map.values()) + 1
        pretrained_VGG19_model = torchvision.models.vgg19(pretrained=True)
        pretrained_VGG19_layers = list(pretrained_VGG19_model.features.children())
        pretrained_VGG19_needed_layers = pretrained_VGG19_layers[:deepest_layer_depth]
        self.features = torch.nn.Sequential(*pretrained_VGG19_needed_layers)
        # Deactivating gradients prevents the unintended accumulation of gradients and, coincidentally, saves memory.
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def forward(self, batch: Tensor) -> Tensor:
        '''
        Evaluates this (partial) VGG-19 network over the given batch of RGB images.  Since the output of this method
        changes based on the contents of the layer index map, it basically serves no purpose.

        Args:
            images: Tensor [B, 3, R, C] of images.

        Returns:
            Tensor [B, D, R, C] of arbitrary feature maps (determined by the deepest named layer).
        '''
        raise NotImplementedError('Class "VGG19" does not implement method "forward".')

    def extract_feature_maps(self, batch: Tensor, layer_names: List[str], normalize: bool = False) -> Dict[str, Tensor]:
        '''
        Extracts the feature maps associated with the given layers in the VGG-19 network from the provided batch.

        Args:
            batch: Tensor [B, 3, R, C] of images.
            layer_names: List of VGG-19 layer names.
            normalize: Boolean indicating if the images in the batch should be normalized.  Enabling this feature
                       tends to negatively influence the learning process of an SVBRDFAutoencoder.

        Returns:
            Dictionary which maps layer names to feature maps (i.e., activations).
        '''
        if normalize:
            # The expected input format of the pretrained VGG-19 network is described in https://pytorch.org/hub/pytorch_vision_vgg/.
            transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            activations = torch.stack([transform(img) for img in batch], dim=0)
        else:
            activations = batch

        layer_index_to_name_map = {self.layer_name_to_index_map[name]: name for name in layer_names}
        feature_maps: Dict[str, Tensor] = {}
        depth = max(layer_index_to_name_map.keys()) + 1
        for i in range(depth):
            activations = self.features[i](activations)
            if i in layer_index_to_name_map:
                name = layer_index_to_name_map[i]
                feature_maps[name] = activations
        return feature_maps


def evaluation_wrapper(function: Callable) -> Callable:
    '''
    Decorator which disables PyTorch gradients and sets each network passed to the given function into evaluation mode.
    Before the wrapper terminates, the networks are restored to their original training states.

    Args:
        function: Function that accepts zero or more torch.nn.Module parameters as input.

    Returns:
        Wrapper which implements the aforementioned behaviour.
    '''
    def wrapper(**kwargs: Any):
        networks = [(network, network.training) for network in kwargs.values() if isinstance(network, torch.nn.Module)]
        with torch.no_grad():
            for network, _ in networks:
                network.eval()
            function(**kwargs)
            for network, training in networks:
                network.train(training)
    return wrapper
