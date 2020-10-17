import datetime
import image
import logging
import network
import shader
import torch.optim
import tqdm
import utils

from dataset import Dataset, Texture
from light import PunctualLight
from network import SVBRDFAutoencoder, VGG19
from svbrdf import SVBRDF
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Iterable, Tuple
from viewer import PerspectiveViewer


# The following type annotations help streamline and unify function signatures.
Losses = Dict[str, Dict[str, Tensor]]
Weights = Dict[str, Dict[str, float]]


def optimize(autoencoder: SVBRDFAutoencoder, svbrdf: SVBRDF, datasets: Dict[str, Dataset], optimizer: torch.optim.Optimizer,  # type: ignore
             epochs: int, cycles: int, samples: int, frequencies: Dict[str, int], loss_weights: Weights, early_stopping: Dict,
             experiment: str) -> None:
    '''
    Optimizes the given SVBRDF autoencoder using the provided SVBRDF, Datasets, Optimizer, and hyperparameters.

    Args:
        autoencoder: SVBRDFAutoencoder to be optimized.
        svbrdf: SVBRDF intended for the output of the SVBRDF autoencoder.
        datasets: Mapping between Dataset names (e.g., "Training") and, well, Datasets.
        optimizer: Optimizer that updates the parameters in the SVBRDF autoencoder.
        epochs: Number of training epochs.
        cycles: Number of training steps to execute during each epoch.
        samples: Size of a training batch.
        frequencies: Mapping between event names (e.g., "Parameter Checkpoint") and the number of training steps between
                     executions of these events.
        loss_weights: Mapping between loss types (e.g., "Reconstruction") and dictionaries that associate loss components
                      (e.g., "Style") with their corresponding weights.
        experiment: Name of the current experiment.
    '''
    # The structure of the given dictionaries are checked here rather than the Configuration to avoid bloating distant
    # code and keep the relevant implementation in one place.
    for key in ('Training', 'Testing'):
        assert key in datasets, f'Dataset dictionary is missing key {key}.'
    for key in ('Reconstruction',):
        assert key in loss_weights, f'Loss weights dictionary is missing key {key}.'
    # for key in ('Content', 'Style'):
    #     assert key in loss_weights['Diversity'], f'Loss weights dictionary is missing key {key} under scope "Diversity".'
    for key in ('Content', 'Style', 'Texel'):
        assert key in loss_weights['Reconstruction'], f'Loss weights dictionary is missing key {key} under scope "Reconstruction".'
    for key in ('Tests Publication', 'Image Publication', 'Parameter Checkpoint'):
        assert key in frequencies, f'Frequencies dictionary is missing key {key}.'
    for key in ('Epsilon', 'Patience'):
        assert key in early_stopping, f'Early stopping dictionary is missing key {key}.'

    def zero_backward_step(loss: Tensor) -> None:
        '''Updates the parameters of the SVBRDF autoencoder using the given loss Tensor, taking care to clear gradients beforehand.'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Sharing the same VGG-19 network instance across all invocations is absolutely critical to performance.
    # Furthermore, the VGG-19 network must be initialized after the primary computation device has been set.
    vgg19 = VGG19().to(utils.get_device_name())
    vgg19.eval()

    # A SummaryWriter can be used to publish data to TensorBoard.
    run_name = datetime.datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S')
    dashboard = SummaryWriter(f'results/runs/{run_name} - {experiment}')

    # Tracking the number of training steps is helpful for TensorBoard tracking and implementing "every-X-step" behaviour.
    steps = 0

    # The best testing loss record tracks the minimum testing loss achieved so far.
    best_testing_loss = torch.tensor(float('inf'))
    # The early stopping counter tracks the number of early stopping checks that have transpired since the best testing loss was updated.
    early_stopping_counter = 0

    # Each epoch introduces a new material to the training loop.
    for epoch in range(epochs):
        for cycle in tqdm.tqdm(range(cycles), desc=f'Epoch {epoch} Progress', total=cycles):
            # Materials from the Dataset are selected in a round-robin fashion following the training technique described
            # in the Diversified Texture Synthesis with Feed-forward Networks paper.
            material = cycle % min(epoch + 1, len(datasets['Training']))

            # The reconstruction loss encourages the network to accurately infer the SVBRDF parameters of a given texture.
            reconstruction_losses = compute_reconstruction_losses(autoencoder=autoencoder, network_svbrdf=svbrdf, samples=samples,
                                                                  dataset=datasets['Training'], material=material, vgg19=vgg19)
            reconstruction_losses['Total'] = loss_weights['Reconstruction']['Content'] * reconstruction_losses['Content'] + \
                                             loss_weights['Reconstruction']['Style'] * reconstruction_losses['Style'] + \
                                             loss_weights['Reconstruction']['Texel'] * reconstruction_losses['Texel']
            zero_backward_step(loss=reconstruction_losses['Total'])

            # The diversity loss encourages the network to encode the style of a texture in the global latent vector.
            # diversity_losses = compute_diversity_losses(autoencoder=autoencoder, network_svbrdf=svbrdf, samples=samples,
            #                                             dataset=datasets['Training'], material=material, vgg19=vgg19)
            # diversity_losses['Total'] = loss_weights['Diversity']['Content'] * diversity_losses['Content'] + \
            #                             loss_weights['Diversity']['Style'] * diversity_losses['Style']
            # zero_backward_step(loss=diversity_losses['Total'])

            # The training losses are published to the dashboard after each training iteration.
            texture = datasets['Training'].textures[material]
            losses = {'Reconstruction': reconstruction_losses}
            publish_scalar_results(dashboard=dashboard, mode='Training', steps=steps, texture=texture, losses=losses)

            # Adding one to the number of steps avoids triggering an event on the first training iteration.
            progress = steps + 1
            if progress % frequencies['Parameter Checkpoint'] == 0:
                autoencoder.save()
            if progress % frequencies['Image Publication'] == 0:
                materials = range(min(epoch + 1, len(datasets['Training'])))
                publish_image_results(dashboard=dashboard, mode='Training', steps=steps, autoencoder=autoencoder,
                                      network_svbrdf=svbrdf, dataset=datasets['Training'], materials=materials)
            if progress % frequencies['Tests Publication'] == 0:
                publish_testing_results(dashboard=dashboard, steps=steps, autoencoder=autoencoder, network_svbrdf=svbrdf,
                                        dataset=datasets['Testing'], samples=samples, loss_weights=loss_weights, vgg19=vgg19)
            if progress % frequencies['Early Stopping'] == 0 and epoch >= len(datasets['Training']):
                # The testing loss is the mean loss of each material in the testing dataset.
                testing_loss = compute_testing_loss(autoencoder=autoencoder, network_svbrdf=svbrdf, dataset=datasets['Testing'],
                                                    samples=samples, loss_weights=loss_weights, vgg19=vgg19)
                # The epsilon factor avoids delaying the early stopping due to noise.
                if testing_loss < best_testing_loss - early_stopping['Epsilon']:
                    best_testing_loss = testing_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping['Patience']:
                        logging.info('Early stopping triggered at epoch %d cycle %d', epoch, cycle)
                        # The parameter weights may no longer be "optimal" but they are probably close enough.
                        autoencoder.save()
                        dashboard.close()
                        return
            steps += 1
    dashboard.close()


@network.evaluation_wrapper
def publish_testing_results(dashboard: SummaryWriter, steps: int, autoencoder: SVBRDFAutoencoder, network_svbrdf: SVBRDF,
                            dataset: Dataset, samples: int, loss_weights: Weights, vgg19: VGG19) -> None:
    '''
    Publishes reconstructions of the materials in the testing Dataset to the given TensorBoard.

    Args:
        dashboard: TensorBoard to host the published data.
        steps: Step count associated with the published data.
        autoencoder: SVBRDFAutoencoder to be tested.
        network_svbrdf: SVBRDF intended for the output of the SVBRDF autoencoder.
        dataset: Dataset consisting of testing inputs and outputs.
        samples: Number of samples to supply to the reconstruction loss function.
        loss_weights: See optimize().
    '''
    materials = [material for material in range(len(dataset))]
    for material in materials:
        texture = dataset.textures[material]
        losses = compute_losses(autoencoder=autoencoder, network_svbrdf=network_svbrdf, dataset=dataset, samples=samples,
                                material=material, loss_weights=loss_weights, vgg19=vgg19)
        publish_scalar_results(dashboard=dashboard, mode='Testing', steps=steps, texture=texture, losses=losses)
    # Visualizing the reconstruction results can yield additional insight beyond the scalar loss data.
    publish_image_results(dashboard=dashboard, mode='Testing', steps=steps, dataset=dataset, materials=materials,
                          autoencoder=autoencoder, network_svbrdf=network_svbrdf)


@network.evaluation_wrapper
def publish_image_results(dashboard: SummaryWriter, mode: str, steps: int, autoencoder: SVBRDFAutoencoder,
                          network_svbrdf: SVBRDF, dataset: Dataset, materials: Iterable[int]) -> None:
    '''
    Publishes a series of images to the given TensorBoard depicting reconstructions (diverse and otherwise) of the
    specified materials.

    Args:
        dashboard: TensorBoard to host the published data.
        mode: Mode associated with the published data (i.e., "Training" or "Testing").
        steps: Step count associated with the published data.
        autoencoder: SVBRDFAutoencoder to be used to reconstruct the Dataset images.
        svbrdf: SVBRDF intended for the output of the SVBRDF autoencoder.
        dataset: Dataset consisting of inputs to the SVBRDF autoencoder and ground-truth Tensors.
        materials: Indices of the materials in the Dataset to be reconstructed.
    '''
    for material in materials:
        texture = dataset.textures[material]
        # One sample should be enough to hint at the reconstruction performance of the SVBRDF autoencoder.
        dataset_batch, (dataset_normals, dataset_svbrdf) = dataset.sample(material)
        network_normals, network_svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.forward(dataset_batch))
        have_radiance, want_radiance = compute_radiance(network_normals=network_normals, network_svbrdf=network_svbrdf,
                                                        dataset_normals=dataset_normals, dataset_svbrdf=dataset_svbrdf)
        # The sRGB colour space applies a desirable gamma correction.
        input_image = image.convert_RGB_to_sRGB(dataset_batch[0, :3].permute(1, 2, 0))
        have_image = image.convert_RGB_to_sRGB(have_radiance[0])
        want_image = image.convert_RGB_to_sRGB(want_radiance[0])
        reconstruction_images = [input_image, want_image, have_image]
        # By convention, the shader module in this repository outputs radiance in [B, R, C, 3] order.
        dashboard.add_images(tag=f'{mode} / {texture}', global_step=steps, dataformats='NHWC', img_tensor=torch.stack(reconstruction_images, dim=0))

        # Two samples are taken for the diversity measure to directly compare the influence of random latent fields.
        # latents = autoencoder.encode(dataset_batch).repeat(2, 1, 1, 1)
        # channels = autoencoder.dimensions['Latent']['Local'][2]
        # latents[:, :channels, :, :] = torch.rand_like(latents[:, :channels, :, :]) * 2 - 1
        # network_normals, network_svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.decode(latents))
        # diversity_radiance, _ = compute_radiance(network_normals=network_normals, network_svbrdf=network_svbrdf,
        #                                          dataset_normals=dataset_normals, dataset_svbrdf=dataset_svbrdf)
        # Don't forget the sRGB conversion.
        # diversity_images = [input_image] + [image.convert_RGB_to_sRGB(radiance) for radiance in diversity_radiance]
        # dashboard.add_images(tag=f'{mode} [Diversity] / {texture}', global_step=steps, dataformats='NHWC', img_tensor=torch.stack(diversity_images, dim=0))


def publish_scalar_results(dashboard: SummaryWriter, mode: str, steps: int, texture: Texture, losses: Losses) -> None:
    '''
    Publishes a series of graphs to the provided TensorBoard showcasing the specified losses.

    Args:
        dashboard: TensorBoard to host the published data.
        mode: Mode associated with the published data (i.e., "Training" or "Testing").
        steps: Step count associated with the published data.
        texture: Texture associated with the published data.
        losses: Mapping between loss types (e.g., "Reconstruction") and dictionaries that associate loss components (e.g.,
                "Style") with their corresponding Tensor values.
    '''
    for key in ('Reconstruction',):
        assert key in losses, f'Loss dictionary is missing key {key}.'
    # for key in ('Content', 'Style'):
    #     assert key in losses['Diversity'], f'Loss dictionary is missing key {key} under scope "Diversity".'
    for key in ('Content', 'Style', 'Texel'):
        assert key in losses['Reconstruction'], f'Loss dictionary is missing key {key} under scope "Reconstruction".'
    # dashboard.add_scalars(main_tag=f'{mode} [Diversity] / {texture}',
    #                       tag_scalar_dict={'Content': float(losses['Diversity']['Content']),
    #                                        'Style': float(losses['Diversity']['Style']),
    #                                        'Total': float(losses['Diversity']['Total'])}, global_step=steps)
    dashboard.add_scalars(main_tag=f'{mode} / {texture}',
                          tag_scalar_dict={'Content': float(losses['Reconstruction']['Content']),
                                           'Style': float(losses['Reconstruction']['Style']),
                                           'Texel': float(losses['Reconstruction']['Texel']),
                                           'Total': float(losses['Reconstruction']['Total'])}, global_step=steps)


@network.evaluation_wrapper
def compute_testing_loss(autoencoder: SVBRDFAutoencoder, network_svbrdf: SVBRDF, dataset: Dataset, samples: int,
                         loss_weights: Weights, vgg19: VGG19) -> Tensor:
    '''
    Computes the mean loss associated with the given SVBRDF autoencoder over the specified testing Dataset.

    Args:
        autoencoder: SVBRDFAutoencoder to be used for the loss calculations.
        network_svbrdf: SVBRDF intended for the output of the SVBRDF autoencoder.
        dataset: (Testing) Dataset to be evaluated.
        samples: See compute_losses().
        loss_weights: See optimize().
        vgg19: See compute_losses().

    Returns:
        Tensor [1] representing the mean combined loss over the testing Dataset.
    '''
    loss = torch.tensor(0.0)
    for material in range(len(dataset)):
        loss += compute_losses(autoencoder=autoencoder, network_svbrdf=network_svbrdf, dataset=dataset, material=material,
                               samples=samples, loss_weights=loss_weights, vgg19=vgg19)['Reconstruction']['Total']
    return loss / len(dataset)


def compute_losses(autoencoder: SVBRDFAutoencoder, network_svbrdf: SVBRDF, dataset: Dataset, material: int, samples: int,
                   loss_weights: Weights, vgg19: VGG19) -> Losses:
    '''
    Computes all of the losses associated with the given SVBRDF autoencoder with respect to the material from the
    specified Dataset.  Note that the losses returned by this function are not suitable for backpropagation.

    Args:
        autoencoder: SVBRDFAutoencoder to be used for the loss calculations.
        network_svbrdf: SVBRDF intended for the output of the SVBRDF autoencoder.
        dataset: Dataset containing the material to be evaluated.
        material: Material in the Dataset to be evaluated.
        samples: Number of samples to supply to the reconstruction loss.
        loss_weights: See optimize().
        vgg19: (Shared) VGG-19 instance to use to compute the reconstruction loss.

    Returns:
        Dictionary containing the reconstruction loss, as expected by publish_scalar_results().
    '''
    reconstruction_losses = compute_reconstruction_losses(autoencoder=autoencoder, network_svbrdf=network_svbrdf, samples=samples,
                                                          dataset=dataset, material=material, vgg19=vgg19)
    reconstruction_losses['Total'] = loss_weights['Reconstruction']['Content'] * reconstruction_losses['Content'] + \
                                     loss_weights['Reconstruction']['Style'] * reconstruction_losses['Style'] + \
                                     loss_weights['Reconstruction']['Texel'] * reconstruction_losses['Texel']

    # diversity_losses = compute_diversity_losses(autoencoder=autoencoder, network_svbrdf=network_svbrdf, samples=samples,
    #                                             dataset=dataset, material=material, vgg19=vgg19)
    # diversity_losses['Total'] = loss_weights['Diversity']['Content'] * diversity_losses['Content'] + \
    #                             loss_weights['Diversity']['Style'] * diversity_losses['Style']

    return {'Reconstruction': reconstruction_losses}


def compute_diversity_losses(autoencoder: SVBRDFAutoencoder, network_svbrdf: SVBRDF, dataset: Dataset, material: int,
                             samples: int, vgg19: VGG19) -> Dict[str, Tensor]:
    '''
    Computes the diversity loss of the given SVBRDF autoencoder with respect to the material from the specified Dataset.
    This loss is meant to train the global and periodic encoders, as well as the decoder of the SVBRDF autoencoder.

    Args:
        autoencoder: SVBRDFAutoencoder to be used for the loss calculation.
        network_svbrdf: SVBRDF intended for the output of the SVBRDF autoencoder.
        dataset: Dataset containing the material to be reconstructed.
        material: Material in the Dataset to be reconstructed.
        samples: Number of random latent fields to create.
        vgg19: (Shared) VGG-19 instance to use to compute the content and style losses.

    Returns:
        Dictionary containing the diversity content and style losses.
    '''
    # Each latent field must be derived from the same sample in case the texture is not a Markov Random Field.
    dataset_batch, (dataset_normals, dataset_svbrdf) = dataset.sample(material)
    latents = autoencoder.encode(dataset_batch).repeat(samples, 1, 1, 1)
    # Replacing the local latent field with a random one should, ideally, cause the reconstructed normal map and SVBRDF
    # parameters to portray a different sample from the same texture.
    channels = autoencoder.dimensions['Latent']['Local'][2]
    latents[:, :channels, :, :] = torch.rand_like(latents[:, :channels, :, :]) * 2 - 1
    network_normals, network_svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.decode(latents))
    # Surface radiance can be interpreted as a projection-corrected rendering of a texture.
    have_radiance, want_radiance = compute_radiance(network_normals=network_normals, network_svbrdf=network_svbrdf,
                                                    dataset_normals=dataset_normals, dataset_svbrdf=dataset_svbrdf)

    # The content loss term encourages the network to reconstruct different spatial features of the input image.
    content_loss = -compute_content_loss(have_radiance=have_radiance, want_radiance=have_radiance.roll(1, 0), vgg19=vgg19)
    # The style loss term encourages the network to reconstruct spatially-independent features of an input image.
    style_loss = compute_style_loss(have_radiance=have_radiance, want_radiance=want_radiance.expand_as(have_radiance), vgg19=vgg19)
    return {'Content': content_loss, 'Style': style_loss}


def compute_reconstruction_losses(autoencoder: SVBRDFAutoencoder, network_svbrdf: SVBRDF, dataset: Dataset, material: int,
                                  samples: int, vgg19: VGG19) -> Dict[str, Tensor]:
    '''
    Computes the reconstruction loss of the given SVBRDF autoencoder with respect to the material from the specified Dataset.
    This loss is meant to train all of the encoders as well as the decoder of the SVBRDF autoencoder.

    Args:
        autoencoder: SVBRDFAutoencoder to be used for the loss calculation.
        network_svbrdf: SVBRDF intended for the output of the SVBRDF autoencoder.
        dataset: Dataset containing the material to be reconstructed.
        material: Material in the Dataset to be reconstructed.
        samples: Number of material samples to take.
        vgg19: (Shared) VGG-19 instance to use to compute the content and style losses.

    Returns:
        Dictionary containing the reconstruction content, style, and texel losses.
    '''
    # Taking multiple samples accelerates training in comparison to SGD.
    dataset_batch, (dataset_normals, dataset_svbrdf) = dataset.sample(material, quantity=samples)
    network_normals, network_svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.forward(dataset_batch))
    # Surface radiance can be interpreted as a projection-corrected rendering of a texture.
    have_radiance, want_radiance = compute_radiance(network_normals=network_normals, network_svbrdf=network_svbrdf,
                                                    dataset_normals=dataset_normals, dataset_svbrdf=dataset_svbrdf)

    # The content loss term encourages the network to reconstruct spatial features of an input image.
    content_loss = compute_content_loss(have_radiance=have_radiance, want_radiance=want_radiance, vgg19=vgg19)
    # The style loss term encourages the network to reconstruct spatially-independent features of an input image.
    style_loss = compute_style_loss(have_radiance=have_radiance, want_radiance=want_radiance, vgg19=vgg19)
    # The texel loss term is a stricter content loss term.
    texel_loss = compute_texel_loss(have_radiance=have_radiance, want_radiance=want_radiance)
    return {'Content': content_loss, 'Style': style_loss, 'Texel': texel_loss}


def compute_radiance(network_normals: Tensor, network_svbrdf: SVBRDF, dataset_normals: Tensor, dataset_svbrdf: SVBRDF) -> Tuple[Tensor, Tensor]:
    '''
    Computes the radiance from the given normals and SVBRDFs with respect to a random point Light and Viewer.

    Args:
        network_normals: Tensor [B, R, C, 3] of SVBRDF autoencoder normals.
        network_svbrdf: SVBRDF with embedded SVBRDF autoencoder parameters.
        dataset_normals: Tensor [B, R, C, 3] of ground-truth normals.
        dataset_svbrdf: SVBRDF with embedded ground-truth parameters.

    Returns:
        Tuple containing the SVBRDF autoencoder and Dataset radiance Tensors.
    '''
    # There is no harm in sharing the same drawing canvas for both the network and dataset renderings.
    texture_rows = dataset_normals.size(1)
    texture_cols = dataset_normals.size(2)
    surface = utils.create_grid(num_rows=texture_rows, num_cols=texture_cols)

    # The Light and Viewer are sampled from a cosine-weighted distribution following the Single-Image SVBRDF Capture
    # with a Rendering-Aware Deep Network paper.
    origin = torch.tensor([0.5, 0.5, 0.0], device=utils.get_device_name())
    lights = [PunctualLight(position=utils.sample_cosine_hemisphere(origin), lumens=torch.rand(1).expand(3) * 2 + 0.5)]
    viewer = PerspectiveViewer(position=utils.sample_cosine_hemisphere(origin))

    network_radiance = shader.shade(surface=surface, normals=network_normals, lights=lights, viewer=viewer, svbrdf=network_svbrdf)
    dataset_radiance = shader.shade(surface=surface, normals=dataset_normals, lights=lights, viewer=viewer, svbrdf=dataset_svbrdf)
    return network_radiance, dataset_radiance


def compute_texel_loss(have_radiance: Tensor, want_radiance: Tensor) -> Tensor:
    '''
    Evaluates the L1 loss between the provided radiance Tensors.

    Args:
        have_radiance: Tensor [B, R, C, 3] of (predicted) radiance values.
        want_radiance: Tensor [B, R, C, 3] of (ground-truth) radiance values.

    Returns:
        L1 loss between the radiance Tensors.
    '''
    return torch.nn.L1Loss(reduction='mean')(have_radiance, want_radiance)


def compute_style_loss(have_radiance: Tensor, want_radiance: Tensor, vgg19: VGG19) -> Tensor:
    '''
    Evaluates the style loss between the provided radiance Tensors as described in the seminal work by Gatys et al.

    Args:
        have_radiance: Tensor [B, R, C, 3] of (predicted) radiance values.
        want_radiance: Tensor [B, R, C, 3] of (ground-truth) radiance values.

    Returns:
        Style loss between the radiance Tensors.
    '''
    style_layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    have_feature_maps = vgg19.extract_feature_maps(have_radiance.permute(0, 3, 1, 2), style_layer_names, True)
    want_feature_maps = vgg19.extract_feature_maps(want_radiance.permute(0, 3, 1, 2), style_layer_names, True)
    loss = torch.tensor(0.0)
    for name in style_layer_names:
        have_Gram_matrices = utils.compute_Gram_matrix(have_feature_maps[name])
        want_Gram_matrices = utils.compute_Gram_matrix(want_feature_maps[name])
        loss += torch.nn.L1Loss(reduction='sum')(have_Gram_matrices, want_Gram_matrices)
    return loss / len(style_layer_names)


def compute_content_loss(have_radiance: Tensor, want_radiance: Tensor, vgg19: VGG19) -> Tensor:
    '''
    Evaluates the content loss between the provided radiance Tensors as described in the Texture Networks paper.

    Args:
        have_radiance: Tensor [B, R, C, 3] of (predicted) radiance values.
        want_radiance: Tensor [B, R, C, 3] of (ground-truth) radiance values.

    Returns:
        Content loss between the radiance Tensors.
    '''
    content_layer_name = 'relu4_2'
    have_feature_maps = vgg19.extract_feature_maps(have_radiance.permute(0, 3, 1, 2), [content_layer_name], True)[content_layer_name]
    want_feature_maps = vgg19.extract_feature_maps(want_radiance.permute(0, 3, 1, 2), [content_layer_name], True)[content_layer_name]
    return torch.nn.L1Loss(reduction='mean')(have_feature_maps, want_feature_maps)
