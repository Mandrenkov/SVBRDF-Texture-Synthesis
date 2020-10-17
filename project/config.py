import image
import torch
import torch.nn
import vector
import utils
import yaml

from camera import Camera, IdentityCamera, PerspectiveCamera
from dataset import Dataset, Texture
from light import Light, DirectionalLight, ImageLight, PunctualLight
from network import SVBRDFAutoencoder
from svbrdf import SVBRDF, BlinnPhongSVBRDF, DisneySVBRDF, LambertianSVBRDF, SubstanceSVBRDF
from torch import Tensor
from transform import Transform, ReflectionTransform, RotationTransform, ElevationTransform, SubstitutionTransform
from typing import Callable, Dict, List, Tuple
from viewer import Viewer, OrthographicViewer, PerspectiveViewer


class Configuration:
    '''
    The Configuration class represents the contents of a YAML configuration file.
    '''

    def __init__(self, path: str) -> None:
        '''
        Constructs a new Configuration from the given configuration file.

        Args:
            path: Path to the YAML configuration file.
        '''
        with open(path, 'r') as file:
            self._config = yaml.safe_load(file)

    def load_album_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, List[Light], Viewer, Camera, Tuple[int, int], int, List[str], str]:
        '''
        Loads the contents of this Configuration assuming a "album" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, Lights, Viewer, and Camera instances from this Configuration, as well as the
            size of the output image, the amount of overlap between the latent tiles, and the paths to the input and
            output files.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Input Paths', 'Output Path', 'Output Size', 'Overlap'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        return autoencoder, svbrdf, lights, viewer, camera, self._config['Flow']['Output Size'], self._config['Flow']['Overlap'], \
               self._config['Flow']['Input Paths'], self._config['Flow']['Output Path']

    def load_blend_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, List[Light], Viewer, Camera, Dict[str, float], List[str], str]:
        '''
        Loads the contents of this Configuration assuming a "blend" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, Light, Viewer, and Camera instances from this Configuration, as well as the paths
            to the input and output files and a mapping between latent components and alpha values.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Input Paths', 'Output Path', 'Alphas'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        return autoencoder, svbrdf, lights, viewer, camera, \
               self._config['Flow']['Alphas'], self._config['Flow']['Input Paths'], self._config['Flow']['Output Path']

    def load_extract_flow(self) -> Tuple[Dataset, Texture, str]:
        '''
        Loads the contents of this Configuration assuming a "extract" flow.

        Returns:
            Dataset containing the desired texture, the Texture itself, and the path to the output file.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Dataset', 'Texture', 'Output Path'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Dataset' in self._config, 'Scope "root" in configuration file is missing key "Dataset".'
        with open(self._config['Dataset'], 'r') as file:
            config = yaml.safe_load(file)
            dataset = Configuration._load_dataset(config, self._config['Flow']['Dataset'])
        return dataset, self._config['Flow']['Texture'], self._config['Flow']['Output Path']

    def load_feedback_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, Camera, Tuple[List[Light], Viewer], Tuple[List[Light], Viewer], str, str, int]:
        '''
        Loads the contents of this Configuration assuming a "feedback" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, and Camera instances from this Configuration, as well as the number of loops and
            the paths to the input and output files.  A pair of Viewer-Light pairs are also returned which communicate
            the desired Light and Viewer settings for the feedback loop and the final rendering, respectively.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Input Path', 'Output Path', 'Loops'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        assert 'Feedback Lights' in self._config, 'Scope "root" in configuration file is missing key "Feedback Lights".'
        feedback_lights = [Configuration._load_light(light) for light in self._config['Feedback Lights']]
        assert 'Feedback Viewer' in self._config, 'Scope "root" in configuration file is missing key "Feedback Viewer".'
        feedback_viewer = Configuration._load_viewer(self._config['Feedback Viewer'])
        assert 'Rendering Lights' in self._config, 'Scope "root" in configuration file is missing key "Rendering Lights".'
        rendering_lights = [Configuration._load_light(light) for light in self._config['Rendering Lights']]
        assert 'Rendering Viewer' in self._config, 'Scope "root" in configuration file is missing key "Rendering Viewer".'
        rendering_viewer = Configuration._load_viewer(self._config['Rendering Viewer'])
        return autoencoder, svbrdf, camera, (feedback_lights, feedback_viewer), (rendering_lights, rendering_viewer), \
               self._config['Flow']['Input Path'], self._config['Flow']['Output Path'], self._config['Flow']['Loops']

    def load_merge_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, List[Light], Viewer, Camera, int, List[str], str]:
        '''
        Loads the contents of this Configuration assuming a "merge" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, Lights, Viewer, and Camera instances from this Configuration, as well as the
            number of overlapping latent "pixels" between the textures and the paths to the input and output files.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Input Paths', 'Output Path', 'Overlap'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        return autoencoder, svbrdf, lights, viewer, camera, \
                self._config['Flow']['Overlap'], self._config['Flow']['Input Paths'], self._config['Flow']['Output Path']

    def load_merl_flow(self) -> Tuple[str, str, str, SVBRDF]:
        '''
        Loads the contents of this Configuration assuming a "merl" flow.

        Returns:
            Path to the MERL 100 BRDF slices image, path to the output file, name of the SciPy optimizer, tolerance of
            the optimizer, and, finally, the SVBRDF instance from this Configuration.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Input Path', 'Output Path', 'Optimizer'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        return self._config['Flow']['Input Path'], self._config['Flow']['Output Path'], self._config['Flow']['Optimizer'], svbrdf

    def load_morph_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, List[Light], Viewer, Camera, int, List[str], str]:
        '''
        Loads the contents of this Configuration assuming a "morph" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, Lights, Viewer, and Camera instances from this Configuration, as well as the
            number of tiles between the textures and the paths to the input and output files.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Input Paths', 'Output Path', 'Between'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        return autoencoder, svbrdf, lights, viewer, camera, \
                self._config['Flow']['Between'], self._config['Flow']['Input Paths'], self._config['Flow']['Output Path']

    def load_mosaic_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, List[Light], Viewer, Camera, int, str, str]:
        '''
        Loads the contents of this Configuration assuming a "mosaic" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, Lights, Viewer, and Camera instances from this Configuration, as well as the
            amount of overlap between the latent tiles and the paths to the input and output files.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Overlap', 'Input Path', 'Output Path'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        return autoencoder, svbrdf, lights, viewer, camera, \
               self._config['Flow']['Overlap'], self._config['Flow']['Input Path'], self._config['Flow']['Output Path']

    def load_relight_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, List[Light], Viewer, Camera, str, str]:
        '''
        Loads the contents of this Configuration assuming a "relight" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, Light, Viewer, and Camera instances from this Configuration, as well as the paths
            to the input and output files.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Input Path', 'Output Path'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        return autoencoder, svbrdf, lights, viewer, camera, self._config['Flow']['Input Path'], self._config['Flow']['Output Path']

    def load_render_flow(self) -> Tuple[Tensor, SVBRDF, List[Light], Viewer, Camera, str]:
        '''
        Loads the contents of this Configuration assuming a "render" flow.

        Returns:
            Tensor [R, C, 3] of normals, the SVBRDF, Light, Viewer, and Camera instances from this Configuration, and
            the path to the output file.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        normals = Configuration._load_normals(self._config['Flow'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        assert 'Output Path' in self._config['Flow'], 'Scope "Flow" in configuration file is missing key "Output".'
        return normals, svbrdf, lights, viewer, camera, self._config['Flow']['Output Path']

    def load_shuffle_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, List[Light], Viewer, Camera, Tuple[int, int], Tuple[int, int], str, str]:
        '''
        Loads the contents of this Configuration assuming a "shuffle" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, Lights, Viewer, and Camera instances from this Configuration, as well as the
            size of the sampled tiles and output texture, in addition to the paths to the input and output files.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Input Path', 'Output Path', 'Output Size', 'Tile Size'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        return autoencoder, svbrdf, lights, viewer, camera, self._config['Flow']['Tile Size'], self._config['Flow']['Output Size'], \
               self._config['Flow']['Input Path'], self._config['Flow']['Output Path']

    def load_tile_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, List[Light], Viewer, Camera, int, str, str]:
        '''
        Loads the contents of this Configuration assuming a "tile" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, Light, Viewer, and Camera instances from this Configuration, as well as the
            number of overlapping latent "pixels" and the paths to the input and output files.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Overlap', 'Input Path', 'Output Path'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        return autoencoder, svbrdf, lights, viewer, camera, self._config['Flow']['Overlap'], \
               self._config['Flow']['Input Path'], self._config['Flow']['Output Path']

    def load_training_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, Dict[str, Dataset], torch.optim.Optimizer, Dict]:  # type: ignore
        '''
        Loads the contents of this Configuration assuming a "training" flow.

        Returns:
            SVBRDFAutoencoder, SVBRDF, training and testing Datasets, and network Optimizer instances from this
            Configuration.  The hyperparameters under the "Flow" scope of the Configuration are also returned.
        '''
        for key in ('Network', 'SVBRDF', 'Dataset', 'Optimizer', 'Flow'):
            assert key in self._config, f'Scope "root" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        with open(self._config['Dataset'], 'r') as file:
            config = yaml.safe_load(file)
            assert 'Tuning' in self._config['Flow'], 'Scope "Flow" in configuration file is missing key "Tuning".'
            if self._config['Flow']['Tuning']:
                # Hyperparameters should be tuned on the validation set rather than the training set to avoid bias.
                datasets = {'Training': Configuration._load_dataset(config, 'Validation'), 'Testing': Configuration._load_dataset(config, 'Testing')}
            else:
                # For similar reasons, testing (throughout the training process) must be performed on the validation set.
                datasets = {'Training': Configuration._load_dataset(config, 'Training'), 'Testing': Configuration._load_dataset(config, 'Validation')}
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        optimizer = Configuration._load_optimizer(self._config['Optimizer'], autoencoder)
        hyperparameters = {}
        for key in ('Experiment', 'Cycles', 'Early Stopping', 'Epochs', 'Frequencies', 'Loss Weights', 'Samples'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
            name = key.lower().replace(' ', '_')
            hyperparameters[name] = self._config['Flow'][key]
        return autoencoder, svbrdf, datasets, optimizer, hyperparameters

    def load_warp_flow(self) -> Tuple[SVBRDFAutoencoder, SVBRDF, List[Light], Viewer, Camera, Tuple[int, int], str, str]:
        '''
        Loads the contents of this Configuration assuming a "warp" flow.

        Returns:
            SVBRDF Autoencoder, SVBRDF, Lights, Viewer, and Camera instances from this Configuration, as well as the
            desired output texture size and the paths to the input and output files.
        '''
        assert 'Flow' in self._config, 'Scope "root" in configuration file is missing key "Flow".'
        for key in ('Input Path', 'Output Path', 'Output Size'):
            assert key in self._config['Flow'], f'Scope "Flow" in configuration file is missing key "{key}".'
        assert 'Network' in self._config, 'Scope "root" in configuration file is missing key "Network".'
        autoencoder = Configuration._load_network(self._config['Network'])
        assert 'SVBRDF' in self._config, 'Scope "root" in configuration file is missing key "SVBRDF".'
        svbrdf = Configuration._load_SVBRDF(self._config['SVBRDF'])
        assert 'Lights' in self._config, 'Scope "root" in configuration file is missing key "Lights".'
        lights = [Configuration._load_light(light) for light in self._config['Lights']]
        assert 'Viewer' in self._config, 'Scope "root" in configuration file is missing key "Viewer".'
        viewer = Configuration._load_viewer(self._config['Viewer'])
        assert 'Camera' in self._config, 'Scope "root" in configuration file is missing key "Camera".'
        camera = Configuration._load_camera(self._config['Camera'])
        return autoencoder, svbrdf, lights, viewer, camera, self._config['Flow']['Output Size'], \
               self._config['Flow']['Input Path'], self._config['Flow']['Output Path']

    @staticmethod
    def _load_normals(config: Dict) -> Tensor:
        '''
        Loads a normal map from the given flow configuration.

        Args:
            config: Flow configuration.
        
        Returns:
            Tensor [1, R, C, 3] of surface normals as specified by the normal map image.
        '''
        assert 'Normals' in config, 'Scope "Flow" in configuration file is missing key "Normals".'
        return vector.normalize(2 * image.load(config['Normals']).unsqueeze(0) - 1)

    @staticmethod
    def _load_SVBRDF(config: Dict) -> SVBRDF:
        '''
        Loads an SVBRDF from the given SVBRDF configuration.

        Args:
            config: SVBRDF configuration.
        
        Returns:
            SVBRDF instance as specified by the SVBRDF configuration.
        '''
        parameters = Tensor()
        if 'Parameters' in config:
            for (i, parameter_config) in enumerate(config['Parameters']):
                for key in ('Type', 'Path'):
                    assert key in parameter_config, f'SVBRDF parameter {i} in configuration file is missing key "{key}".'
                parameter = image.load(parameter_config['Path'], parameter_config['Type']).unsqueeze(0)
                parameters = parameter if parameters.size(0) == 0 else torch.cat([parameters, parameter], dim=3)

        assert 'Type' in config, 'Scope "SVBRDF" in configuration file is missing key "Type".'
        if config['Type'] == 'Lambertian':
            return LambertianSVBRDF(parameters)
        elif config['Type'] == 'Blinn-Phong':
            return BlinnPhongSVBRDF(parameters)
        elif config['Type'] == 'Substance':
            return SubstanceSVBRDF(parameters)
        elif config['Type'] == 'Disney':
            return DisneySVBRDF(parameters)
        else:
            raise ValueError('SVBRDF type "%s" is not supported.' % config['Type'])

    @staticmethod
    def _load_light(config: Dict) -> Light:
        '''
        Loads a Light from the given Light configuration.

        Args:
            config: Light configuration.
        
        Returns:
            Light instance as specified by the Light configuration.
        '''
        assert 'Type' in config, 'Scope "Light" in configuration file is missing key "Type".'
        if config['Type'] == 'Directional':
            for key in ('Direction', 'Lumens'):
                assert key in config, f'Scope "Light" in configuration file is missing key "{key}".'
            return DirectionalLight(direction=torch.tensor(config['Direction'], dtype=torch.float, device=utils.get_device_name()),
                                    lumens=torch.tensor(config['Lumens'], dtype=torch.float, device=utils.get_device_name()))
        elif config['Type'] == 'Punctual':
            for key in ('Position', 'Lumens'):
                assert key in config, f'Scope "Light" in configuration file is missing key "{key}".'
            return PunctualLight(position=torch.tensor(config['Position'], dtype=torch.float, device=utils.get_device_name()),
                                 lumens=torch.tensor(config['Lumens'], dtype=torch.float, device=utils.get_device_name()))
        elif config['Type'] == 'Image':
            for key in ('Path', 'Samples', 'Intensity'):
                assert key in config, f'Scope "Light" in configuration file is missing key "{key}".'
            return ImageLight(path=config['Path'], num_samples=config['Samples'], intensity=config['Intensity'])
        else:
            raise ValueError('Light type "%s" is not supported.' % config['Type'])

    @staticmethod
    def _load_viewer(config: Dict) -> Viewer:
        '''
        Loads a Viewer from the given Viewer configuration.

        Args:
            config: Viewer configuration.
        
        Returns:
            Viewer instance as specified by the Viewer configuration.
        '''
        assert 'Type' in config, 'Scope "Viewer" in configuration file is missing key "Type".'
        if config['Type'] == 'Orthographic':
            assert 'Direction' in config, 'Scope "Viewer" in configuration file is missing key "Direction".'
            return OrthographicViewer(direction=torch.tensor(config['Direction'], dtype=torch.float, device=utils.get_device_name()))
        elif config['Type'] == 'Perspective':
            assert 'Position' in config, 'Scope "Viewer" in configuration file is missing key "Position".'
            return PerspectiveViewer(position=torch.tensor(config['Position'], dtype=torch.float, device=utils.get_device_name()))
        else:
            raise ValueError('Viewer type "%s" is not supported.' % config['Type'])

    @staticmethod
    def _load_camera(config: Dict) -> Camera:
        '''
        Loads a Camera from the given Camera configuration.

        Args:
            config: Camera configuration.
        
        Returns:
            Camera instance as specified by the Camera configuration.
        '''
        assert 'Type' in config, 'Scope "Camera" in configuration file is missing key "Type".'
        if config['Type'] == 'Identity':
            return IdentityCamera()
        if config['Type'] == 'Perspective':
            for key in ('Position', 'Direction', 'Field of View', 'Resolution', 'Exposure'):
                assert key in config, f'Scope "Camera" in configuration file is missing key "{key}".'
            return PerspectiveCamera(position=torch.tensor(config['Position'], dtype=torch.float, device=utils.get_device_name()),
                                     direction=torch.tensor(config['Direction'], dtype=torch.float, device=utils.get_device_name()),
                                     field_of_view=torch.tensor(config['Field of View'], dtype=torch.float, device=utils.get_device_name()),
                                     resolution=torch.tensor(config['Resolution'], device=utils.get_device_name()),
                                     exposure=config['Exposure'])
        else:
            raise ValueError('Camera type "%s" is not supported.' % config['Type'])

    @staticmethod
    def _load_network(path: str) -> SVBRDFAutoencoder:
        '''
        Loads an SVBRDFAutoencoder from the given SVBRDFAutoencoder configuration.

        Args:
            path: Path to the SVBRDFAutoencoder YAML configuration.
        
        Returns:
            SVBRDFAutoencoder instance as specified by the SVBRDFAutoencoder configuration.
        '''
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        for key in ('Dimensions', 'Parameters', 'Encoders', 'Decoder'):
            assert key in config, f'Scope "root" in configuration file is missing key "{key}".'
        for key in ('Path', 'Load'):
            assert key in config['Parameters'], f'Scope "Parameters" in configuration file is missing key "{key}".'
        path = config['Parameters']['Path']
        load = config['Parameters']['Load']
        dims = config['Dimensions']
        encoders = {}
        for key in ('Local', 'Global', 'Periodic'):
            encoders[key] = Configuration._load_subnetwork(config=config['Encoders'][key])
        decoder = Configuration._load_subnetwork(config=config['Decoder'])
        device = utils.get_device_name()
        autoencoder = SVBRDFAutoencoder(dims=dims, path=path, encoders=encoders, decoder=decoder).to(device)
        if load:
            autoencoder.load()
        return autoencoder

    @staticmethod
    def _load_optimizer(config: Dict, network: torch.nn.Module) -> torch.optim.Optimizer:  # type: ignore
        '''
        Loads an Optimizer for the provided neural network from the given Optimizer configuration.

        Args:
            config: Optimizer configuration.
            network: Neural network to optimize.
        
        Returns:
            Optimizer instance over the parameters of the given neural network, as specified by the Optimizer configuration.
        '''
        assert 'Type' in config, f'Scope "Optimizer" in configuration file is missing key "Type".'
        if config['Type'] == 'Adam':
            for key in ('Learning Rate', 'Betas', 'Weight Decay'):
                assert key in config, f'Scope "Optimizer" in configuration file is missing key "{key}".'
            return torch.optim.Adam(network.parameters(), lr=config['Learning Rate'], betas=config['Betas'], weight_decay=config['Weight Decay'])
        else:
            raise ValueError('Optimizer type "%s" is not supported.' % config['Type'])

    @staticmethod
    def _load_subnetwork(config: Dict) -> torch.nn.Module:
        '''
        Loads a neural network from the given neural network configuration.

        Args:
            config: Neural network configuration.
        
        Returns:
            PyTorch Module instance as specified by the neural network configuration.
        '''
        for key in ('Layers', 'Initialization'):
            assert key in config, f'Network in configuration file is missing key "{key}".'
        network = torch.nn.Sequential(*[Configuration._load_subnetwork_layer(layer) for layer in config['Layers']])
        initializer = Configuration._load_subnetwork_initializer(config['Initialization'])
        network.apply(initializer)
        return network

    @staticmethod
    def _load_subnetwork_layer(config: Dict) -> torch.nn.Module:
        '''
        Loads a neural network layer from the given neural network layer configuration.

        Args:
            config: Neural network layer configuration.
        
        Returns:
            PyTorch Module instance as specified by the neural network layer configuration.
        '''
        assert 'Type' in config, 'Layer in configuration file is missing key "Type".'
        if config['Type'] == 'Convolutional':
            return torch.nn.Conv2d(in_channels=config['Channels'][0], out_channels=config['Channels'][1], kernel_size=config['Kernel'], stride=config['Stride'], padding=config['Kernel'] // 2)
        elif config['Type'] == 'Transpose Convolutional':
            return torch.nn.ConvTranspose2d(in_channels=config['Channels'][0], out_channels=config['Channels'][1], kernel_size=config['Kernel'], stride=config['Stride'], padding=config['Kernel'] // 2, output_padding=1)
        elif config['Type'] == 'Fully Connected':
            return torch.nn.Linear(in_features=config['Features'][0], out_features=config['Features'][1])
        elif config['Type'] == 'Upsample':
            return torch.nn.Upsample(scale_factor=config['Scale'], mode=config['Mode'])
        elif config['Type'] == 'Batch Normalization':
            return torch.nn.BatchNorm2d(num_features=config['Features'])
        elif config['Type'] == 'Flat':
            return torch.nn.Flatten()  # type: ignore
        elif config['Type'] == 'Activation':
            if config['Function'] == 'Leaky ReLU':
                assert 'Slope' in config, 'Leaky ReLU activation in configuration file is missing key "Slope".'
                return torch.nn.LeakyReLU(config['Slope'])
            elif config['Function'] == 'ReLU':
                return torch.nn.ReLU()
            elif config['Function'] == 'SELU':
                return torch.nn.SELU()
            elif config['Function'] == 'Tanh':
                return torch.nn.Tanh()
            elif config['Function'] == 'Sigmoid':
                return torch.nn.Sigmoid()
            else:
                raise ValueError('Activation function "%s" is not supported.' % config['Function'])
        else:
            raise ValueError('Layer type "%s" is not supported.' % config['Type'])

    @staticmethod
    def _load_subnetwork_initializer(config: Dict) -> Callable[[torch.nn.Module], None]:
        '''
        Loads a neural network initializer from the given neural network initializer configuration.

        Args:
            config: Neural network initializer configuration.

        Returns:
            Callback which initializes the weights of a neural network according to the neural network initializer configuration.
        '''
        assert 'Type' in config, f'Network initializer in configuration file is missing key "Type".'
        if config['Type'] == 'Normal':
            def initializer(layer: torch.nn.Module) -> None:
                if any(isinstance(layer, kind) for kind in (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
                    torch.nn.init.normal_(layer.weight, mean=config['Mean'], std=config['Stdev'])
                    torch.nn.init.constant_(layer.bias, val=0.01)
            return initializer
        else:
            raise ValueError('Network initializer type "%s" is not supported.' % config['Type'])

    @staticmethod
    def _load_dataset(config: Dict, name: str) -> Dataset:
        '''
        Loads a testing, training, or validation Dataset from the given Dataset configuration.

        Args:
            config: Dataset configuration.
            name: Name of the Dataset (i.e., "Testing", "Training", or "Validation").

        Returns:
            Dataset instance, as specified by the Dataset configuration and the given name.
        '''
        for key in ('Transforms', 'Dimensions', 'Path', 'Layout', 'SVBRDF', f'{name} Textures', 'Testing Textures', 'Lights', 'Viewer'):
            assert key in config, f'Scope "Dataset" in configuration file is missing key "{key}".'
        svbrdf = Configuration._load_SVBRDF(config['SVBRDF'])
        viewer = Configuration._load_viewer(config['Viewer'])
        lights = [Configuration._load_light(light) for light in config['Lights']]
        transforms = [Configuration._load_transform(transform) for transform in config['Transforms']]
        textures = [Configuration._load_texture(texture) for texture in config[f'{name} Textures']]
        return Dataset(dims=config['Dimensions'], path=config['Path'], layout=config['Layout'],
                       svbrdf=svbrdf, textures=textures, transforms=transforms, lights=lights, viewer=viewer)

    @staticmethod
    def _load_transform(config: Dict) -> Transform:
        '''
        Loads a Transform from the given Transform configuration.

        Args:
            config: Transform configuration.
        
        Returns:
            Transform instance as specified by the Transform configuration.
        '''
        assert 'Type' in config, 'Transform in configuration file is missing key "Type".'
        if config['Type'] == 'Reflection':
            return ReflectionTransform()
        elif config['Type'] == 'Rotation':
            return RotationTransform()
        elif config['Type'] == 'Elevation':
            for key in ('Min Scalar', 'Max Scalar'):
                assert key in config, f'Transform in configuration file is missing key "{key}".'
            return ElevationTransform(min_scalar=config['Min Scalar'], max_scalar=config['Max Scalar'])
        elif config['Type'] == 'Substitution':
            return SubstitutionTransform(targets=config['Targets'])
        else:
            raise ValueError('Transform type "%s" is not supported.' % config['Type'])

    @staticmethod
    def _load_texture(config: Dict) -> Texture:
        '''
        Loads a Texture from the given Texture configuration.

        Args:
            config: Texture configuration.
        
        Returns:
            Texture instance as specified by the Texture configuration.
        '''
        for key in ('Name', 'Category'):
            assert key in config, f'Texture in configuration file is missing key "{key}".'
        return Texture(category=config['Category'], name=config['Name'])
