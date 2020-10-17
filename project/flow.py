import image
import logging
import merl
import shader
import torch
import tqdm
import training
import utils

from camera import Camera
from config import Configuration
from light import Light
from network import SVBRDFAutoencoder
from svbrdf import SVBRDF
from torch import Tensor
from typing import List
from viewer import Viewer


def execute(flow: str, config: Configuration) -> None:
    '''
    Executes the given flow using the provided Configuration.

    Args:
        flow: Name of the flow to execute (e.g., "render").
        config: Configuration to supply to the flow.
    '''
    if flow == 'album':
        _album_flow(config)
    elif flow == 'blend':
        _blend_flow(config)
    elif flow == 'extract':
        _extract_flow(config)
    elif flow == 'feedback':
        _feedback_flow(config)
    elif flow == 'merge':
        _merge_flow(config)
    elif flow == 'merl':
        _merl_flow(config)
    elif flow == 'morph':
        _morph_flow(config)
    elif flow == 'mosaic':
        _mosaic_flow(config)
    elif flow == 'relight':
        _relight_flow(config)
    elif flow == 'render':
        _render_flow(config)
    elif flow == 'shuffle':
        _shuffle_flow(config)
    elif flow == 'tile':
        _tile_flow(config)
    elif flow == 'training':
        _training_flow(config)
    elif flow == 'warp':
        _warp_flow(config)
    else:
        raise Exception(f'Flow "{flow}" is not supported.')


def _album_flow(config: Configuration) -> None:
    '''
    The "album" flow generates an image by blending the latent fields of a random sample of input images.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, lights, viewer, camera, output_size, overlap, input_paths, output_path = config.load_album_flow()
        autoencoder.eval()

        # Interpreting the indexing of dimensions exactly once saves more minutes of debugging than keystrokes.
        num_output_rows = output_size[0]
        num_output_cols = output_size[1]
        num_texture_input_rows = autoencoder.dimensions['Texture']['Input'][0]
        num_texture_input_cols = autoencoder.dimensions['Texture']['Input'][1]
        num_texture_output_rows = autoencoder.dimensions['Texture']['Output'][0]
        num_texture_output_cols = autoencoder.dimensions['Texture']['Output'][1]

        # The number of rows and columns that constitute the latent grid (to be decoded into the output image) can be
        # inferred from the relative size of the output image and the SVBRDF autoencoder input.
        num_grid_rows = num_output_rows // num_texture_output_rows
        num_grid_cols = num_output_cols // num_texture_output_cols
        assert (num_output_rows % num_texture_output_rows == 0) and (num_output_cols % num_texture_output_cols == 0), \
               'SVBRDF autoencoder output size must divide output image size.'

        # The images to be included in the latent grid are chosen uniformly at random with replacement from the specified input images.
        input_images = torch.stack([image.load(path=input_path, encoding='sRGB') for input_path in input_paths], dim=0)
        album_images = input_images[torch.randint(low=0, high=len(input_images) - 1, size=(num_grid_rows * num_grid_cols,))]

        # Before feeding the images through the SVBRDF autoencoder, they must be augmented with a radial distance field.
        input_distance = utils.create_radial_distance_field(num_rows=num_texture_input_rows, num_cols=num_texture_input_cols)
        input_batch = torch.cat([album_images, input_distance.expand(album_images.size(0), -1, -1, -1)], dim=3).permute(0, 3, 1, 2)

        # The latent field is assembled by splitting the batch of latent tiles according to their latent grid rows and
        # then interpolating the latent field between each tile in a bilinear fashion.
        texture_latents = torch.stack(autoencoder.encode(input_batch).permute(0, 2, 3, 1).split(num_grid_cols, dim=0), dim=0)
        blended_latents = utils.interpolate(torch.stack([utils.interpolate(texture_latent_row, overlap=overlap).transpose(0, 1) for texture_latent_row in texture_latents], dim=0), overlap=overlap).transpose(0, 1).unsqueeze(0).permute(0, 3, 1, 2)

        # The previous blending procedure leaves the periodic latent component out of alignment with the field indices.
        channels = {key: autoencoder.dimensions['Latent'][key][2] for key in ('Local', 'Global', 'Periodic')}
        global_field = blended_latents[:, channels['Local']:channels['Local'] + channels['Global'], :, :]
        blended_latents[:, -channels['Periodic']:, :, :] = autoencoder.derive_periodic_field(global_field)

        # The fully-convolutional nature of the SVBRDF decoder trivializes the creation of textures with arbitrary sizes.
        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.decode(blended_latents))
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _blend_flow(config: Configuration) -> None:
    '''
    The "blend" flow blends two textures using a (trained) SVBRDF autoencoder and renders the result.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, lights, viewer, camera, alphas, input_paths, output_path = config.load_blend_flow()
        for key in ('Local', 'Global', 'Periodic'):
            assert key in alphas, f'Alphas dictionary is missing key "{key}".'
            assert 0 <= alphas[key] <= 1, f'Alpha value for key "{key}" falls outside the closed interval [0, 1].'
        autoencoder.eval()

        # It is assumed that the dimensions of the input images will be accepted by the network.
        input_images = torch.stack([image.load(path=input_path, encoding='sRGB') for input_path in input_paths], dim=0)

        # The radial distance field should be the same for both input images.
        num_texture_rows = autoencoder.dimensions['Texture']['Input'][0]
        num_texture_cols = autoencoder.dimensions['Texture']['Input'][1]
        input_distance = utils.create_radial_distance_field(num_rows=num_texture_rows, num_cols=num_texture_cols)
        # By convention, PyTorch expects Tensors to be in [B, D, R, C] format.
        input_batch = torch.cat([input_images, input_distance.expand(2, -1, -1, -1)], dim=3).permute(0, 3, 1, 2)

        # The blended latent tensor must have a batch dimension to proceed through the SVBRDF decoder.
        texture_latents = autoencoder.encode(input_batch)
        blended_latents = torch.zeros_like(texture_latents[:1])
        start_channel = 0
        for key in ('Local', 'Global', 'Periodic'):
            # Crucially, the latent components must be traversed in smallest-to-greatest-depth order.
            step_channel = autoencoder.dimensions['Latent'][key][2]
            stop_channel = start_channel + step_channel
            channels = slice(start_channel, stop_channel)
            # An alpha value of 0 represents the first texture while an alpha value of 1 represents the second texture.
            blended_latents[0, channels, :, :] = texture_latents[0, channels, :, :] * (1 - alphas[key]) + \
                                                 texture_latents[1, channels, :, :] * alphas[key]
            start_channel = stop_channel
        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.decode(blended_latents))
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _extract_flow(config: Configuration) -> None:
    '''
    The "extract" flow renders a flash-lit picture of a texture from the dataset.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        dataset, texture, output = config.load_extract_flow()
        material = dataset.textures.index(texture)
        batch, _ = dataset.sample(material)
        picture = batch[0, :3].permute(1, 2, 0)
        image.save(path=output, image=picture, encoding='sRGB')


def _feedback_flow(config: Configuration) -> None:
    '''
    The "feedback" flow iteratively infers the SVBRDF parameters of a texture, renders it, and feeds the output of the
    rendering back into the network.  The purpose of this flow is to test the robustness of an SVBRDF autoencoder.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, camera, (feedback_lights, feedback_viewer), (rendering_lights, rendering_viewer), input_path, output_path, loops = config.load_feedback_flow()
        autoencoder.eval()

        # It is assumed that the dimensions of the input image will be accepted by the network.
        input_image = image.load(path=input_path, encoding='sRGB')
        num_texture_rows = input_image.size(0)
        num_texture_cols = input_image.size(1)
        input_distance = utils.create_radial_distance_field(num_rows=num_texture_rows, num_cols=num_texture_cols)

        # By convention, PyTorch expects Tensors to be in [B, D, R, C] format.
        input_batch = torch.cat([input_image, input_distance], dim=2).unsqueeze(0).permute(0, 3, 1, 2)

        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.forward(input_batch))
        surface = utils.create_grid(num_rows=num_texture_rows, num_cols=num_texture_cols)

        for i in tqdm.tqdm(range(loops), desc='Feedback Looping'):
            # The slightly-awkward ordering of statements before and inside the loops ensures that |loops| can be set to zero.
            input_image = shader.shade(surface=surface, normals=normals, lights=feedback_lights, viewer=feedback_viewer, svbrdf=svbrdf)[0]
            input_batch = torch.cat([input_image, input_distance], dim=2).unsqueeze(0).permute(0, 3, 1, 2)
            normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.forward(input_batch))
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=rendering_lights, viewer=rendering_viewer, camera=camera, path=output_path)


def _merge_flow(config: Configuration) -> None:
    '''
    The "merge" flow melds two overlapping textures by smoothly blending their latent fields.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, lights, viewer, camera, overlap, input_paths, output_path = config.load_merge_flow()
        autoencoder.eval()

        # It is assumed that the dimensions of the input images will be accepted by the network.
        input_images = torch.stack([image.load(path=input_path, encoding='sRGB') for input_path in input_paths], dim=0)
        # The radial distance field should be the same for both input images.
        num_texture_rows = autoencoder.dimensions['Texture']['Input'][0]
        num_texture_cols = autoencoder.dimensions['Texture']['Input'][1]
        input_distance = utils.create_radial_distance_field(num_rows=num_texture_rows, num_cols=num_texture_cols)
        # By convention, PyTorch expects Tensors to be in [B, D, R, C] format.
        input_batch = torch.cat([input_images, input_distance.expand(2, -1, -1, -1)], dim=3).permute(0, 3, 1, 2)

        # The width and height of the SVBRDF autoencoder latent are shared between all latent components.
        num_latent_rows = autoencoder.dimensions['Latent']['Local'][0]
        num_latent_cols = autoencoder.dimensions['Latent']['Local'][1]

        # The latent field corresponding to each texture must be padded in the region where it has no influence.
        device = utils.get_device_name()
        channels = {key: autoencoder.dimensions['Latent'][key][2] for key in ('Local', 'Global', 'Periodic')}
        padding = torch.zeros((num_latent_rows, num_latent_cols - overlap, sum(channels.values())), device=device)

        # The latent field is blended smoothly across the overlapping region as follows:
        #     +------------+---------------------+------------+
        #     |  α = 0.00  |  α = 0.00 ... 1.00  |  α = 1.00  |
        #     +------------+---------------------+------------+
        #                   <----- Overlap ----->
        texture_latents = autoencoder.encode(input_batch).permute(0, 2, 3, 1)
        widened_latents = torch.stack([torch.cat([texture_latents[0], padding], dim=1),
                                       torch.cat([padding, texture_latents[1]], dim=1)], dim=0)
        alphas = torch.cat([torch.zeros(num_latent_cols - overlap, device=device),
                            torch.linspace(0, 1, overlap, device=device),
                            torch.ones(num_latent_cols - overlap, device=device)]).expand(num_latent_rows, -1).unsqueeze(-1)
        blended_latents = torch.lerp(widened_latents[0], widened_latents[1], alphas).permute(2, 0, 1)

        # The periodic component should be replaced to be consistent with the blended global field..
        global_field = blended_latents[channels['Local']:channels['Local'] + channels['Global'], :, :]
        blended_latents[-channels['Periodic']:, :, :] = autoencoder.derive_periodic_field(global_field.unsqueeze(0)).squeeze(0)

        # The fully-convolutional nature of the SVBRDF decoder trivializes the creation of textures with arbitrary sizes.
        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.decode(blended_latents.unsqueeze(0)))
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _merl_flow(config: Configuration) -> None:
    '''
    The "merl" flow fits an SVBRDF to each BRDF slice from the MERL 100 dataset.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    if torch.backends.cudnn.enabled:  # type: ignore
        # Optimization is always faster on the CPU due to device transfer latency.
        torch.backends.cudnn.enabled = False  # type: ignore
        torch.set_default_tensor_type(torch.FloatTensor)
        logging.warning('Disabled CUDA due to device transfer latency')
    input_path, output_path, optimizer, svbrdf = config.load_merl_flow()
    merl.fit(input_path=input_path, output_path=output_path, optimizer=optimizer, svbrdf=svbrdf)


def _morph_flow(config: Configuration) -> None:
    '''
    The "morph" flow morphs one texture into another over a series of discrete tiles.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, lights, viewer, camera, between, input_paths, output_path = config.load_morph_flow()
        autoencoder.eval()

        # The total number of tiles includes the two textures on either end as well as the tiles between the textures.
        tiles = 2 + between
        device = utils.get_device_name()

        # It is assumed that the dimensions of the input images will be accepted by the network.
        input_images = torch.stack([image.load(path=input_path, encoding='sRGB') for input_path in input_paths], dim=0)
        # The radial distance field should be the same for both input images.
        num_texture_rows = autoencoder.dimensions['Texture']['Input'][0]
        num_texture_cols = autoencoder.dimensions['Texture']['Input'][1]
        input_distance = utils.create_radial_distance_field(num_rows=num_texture_rows, num_cols=num_texture_cols)
        # By convention, PyTorch expects Tensors to be in [B, D, R, C] format.
        input_batch = torch.cat([input_images, input_distance.expand(2, -1, -1, -1)], dim=3).permute(0, 3, 1, 2)

        # The width and height of the SVBRDF autoencoder latent are shared between all latent components.
        num_latent_rows = autoencoder.dimensions['Latent']['Local'][0]
        num_latent_cols = autoencoder.dimensions['Latent']['Local'][1]

        # The local field latent is blended such that each texel within a tile has the same alpha component.
        #     +------------+------------+------------+------------+------------+
        #     |  α = 0.00  |  α = 0.25  |  α = 0.50  |  α = 0.75  |  α = 1.00  |
        #     +------------+------------+------------+------------+------------+
        local_encoder_output = autoencoder.encoders['Local'].forward(input_batch)
        local_field_output = local_encoder_output.repeat(1, 1, 1, tiles).permute(0, 2, 3, 1)
        local_field_alphas = torch.linspace(0, 1, tiles, device=device).repeat_interleave(num_latent_cols).expand(num_latent_rows, -1).unsqueeze(-1)
        local_field = torch.lerp(local_field_output[0], local_field_output[1], local_field_alphas).permute(2, 0, 1)

        # The global field latent is blended continuously between the left and right textures.
        #     +------------+------------+------------+------------+------------+
        #     |  α = 0.00  |  α = 0.00 ... ... 0.50 ... ... 1.00  |  α = 1.00  |
        #     +------------+------------+------------+------------+------------+
        global_encoder_output = autoencoder.encoders['Global'].forward(input_batch)
        global_field_output = global_encoder_output.expand(num_latent_rows, num_latent_cols * tiles, -1, -1).permute(2, 0, 1, 3)
        global_field_alphas = torch.cat([torch.zeros(num_latent_cols, device=device),
                                         torch.linspace(0, 1, num_latent_cols * between, device=device),
                                         torch.ones(num_latent_cols, device=device)]).expand(num_latent_rows, -1).unsqueeze(-1)
        global_field = torch.lerp(global_field_output[0], global_field_output[1], global_field_alphas).permute(2, 0, 1)

        # Fortunately, the periodic field latent does not demand any special treatment.
        periodic_field = autoencoder.derive_periodic_field(global_field.unsqueeze(0)).squeeze(0)

        # The fully-convolutional nature of the SVBRDF decoder trivializes the creation of textures with arbitrary sizes.
        latents = torch.cat([local_field, global_field, periodic_field], dim=0).unsqueeze(0)
        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.decode(latents))
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _mosaic_flow(config: Configuration) -> None:
    '''
    The "mosaic" flow reconstructs an image of arbitrary scale by partitioning the given image into smaller images,
    encoding the smaller images as latent fields, and then blending the resulting latent fields in a bilinear fashion.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, lights, viewer, camera, overlap, input_path, output_path = config.load_mosaic_flow()
        autoencoder.eval()

        input_image = image.load(path=input_path, encoding='sRGB')

        # If the input size of the SVBRDF autoencoder does not evenly divide the input image, no valid partitioning exists.
        num_image_rows = input_image.size(0)
        num_image_cols = input_image.size(1)
        num_texture_rows = autoencoder.dimensions['Texture']['Input'][0]
        num_texture_cols = autoencoder.dimensions['Texture']['Input'][1]
        assert (num_image_rows % num_texture_rows == 0) and (num_image_cols % num_texture_cols == 0), 'SVBRDF autoencoder input size must divide input image size.'

        # The input batch is constructed by splitting the image and distance fragments by row and then by column, and
        # then concatenating the result in such a way as to form a single column which can be stacked to form a batch.
        input_distance = utils.create_radial_distance_field(num_rows=num_image_rows, num_cols=num_image_cols)
        input_tensor = torch.cat([input_image, input_distance], dim=2)
        input_batch = torch.cat([torch.stack([input_batch_tile for input_batch_tile in input_batch_row.split(num_texture_cols, dim=1)], dim=0) for input_batch_row in input_tensor.split(num_texture_rows, dim=0)], dim=0).permute(0, 3, 1, 2)

        # The latent dimensions defining the size of a sample latent field can be visualized as follows:
        #   +------------+------------+------------+------------+        +----+----+----+----+
        #   |            |            |            |            |        |  1 |  2 |  3 |  4 |
        #   |      1     |      2     |      3     |      4     |        +----+----+----+----+
        #   |            |            |            |            |        |  5 |  6 |  7 |  8 |
        #   +------------+------------+------------+------------+        +----+----+----+----+
        #   |            |            |            |            |        |  9 | 10 | 11 | 12 |
        #   |      5     |      6     |      7     |      8     |        +----+----+----+----+
        #   |            |            |            |            |         <----- Grid ------>
        #   +------------+------------+------------+------------+
        #   |            |            |            |            |
        #   |      9     |     10     |     11     |     12     |
        #   |            |            |            |            |
        #   +------------+------------+------------+------------+
        #    <-- Tile -->
        #    <----------------- Latent Field ------------------>
        num_grid_cols = num_image_cols // num_texture_cols

        # The latent field is assembled by splitting the batch of latent tiles according to their latent grid rows and
        # then interpolating the latent field between each tile in a bilinear fashion.
        texture_latents = torch.stack(autoencoder.encode(input_batch).permute(0, 2, 3, 1).split(num_grid_cols, dim=0), dim=0)
        blended_latents = utils.interpolate(torch.stack([utils.interpolate(texture_latent_row, overlap=overlap).transpose(0, 1) for texture_latent_row in texture_latents], dim=0), overlap=overlap).transpose(0, 1).unsqueeze(0).permute(0, 3, 1, 2)

        # The previous blending procedure leaves the periodic latent component out of alignment with the field indices.
        channels = {key: autoencoder.dimensions['Latent'][key][2] for key in ('Local', 'Global', 'Periodic')}
        global_field = blended_latents[:, channels['Local']:channels['Local'] + channels['Global'], :, :]
        blended_latents[:, -channels['Periodic']:, :, :] = autoencoder.derive_periodic_field(global_field)

        # The fully-convolutional nature of the SVBRDF decoder trivializes the creation of textures with arbitrary sizes.
        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.decode(blended_latents))
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _relight_flow(config: Configuration) -> None:
    '''
    The "relight" flow renders a picture of a texture using a Light, Viewer, and Camera from a (trained) SVBRDF autoencoder.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, lights, viewer, camera, input_path, output_path = config.load_relight_flow()
        autoencoder.eval()
        # It is assumed that the dimensions of the input image will be accepted by the network.
        input_image = image.load(path=input_path, encoding='sRGB')
        num_texture_rows = input_image.size(0)
        num_texture_cols = input_image.size(1)
        input_distance = utils.create_radial_distance_field(num_rows=num_texture_rows, num_cols=num_texture_cols)
        # By convention, PyTorch expects Tensors to be in [B, D, R, C] format.
        input_batch = torch.cat([input_image, input_distance], dim=2).unsqueeze(0).permute(0, 3, 1, 2)
        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.forward(input_batch))
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _render_flow(config: Configuration) -> None:
    '''
    The "render" flow renders a picture of a texture using a Light, Viewer, Camera, and a set of SVBRDF parameter maps.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        normals, svbrdf, lights, viewer, camera, output_path = config.load_render_flow()
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _shuffle_flow(config: Configuration) -> None:
    '''
    The "shuffle" flow expands the SVBRDF parameters of an image to fill an arbitrary plane by shuffling latent tiles.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, lights, viewer, camera, tile_size, output_size, input_path, output_path = config.load_shuffle_flow()
        autoencoder.eval()

        # Continuing to index sizes with 0 and 1 is simultaneously confusing and a potential debugging nightmare.
        num_tile_rows, num_tile_cols = tile_size
        num_output_rows, num_output_cols = output_size

        # Similarly, it is worthwhile to give names to the otherwise-generic SVBRDF autoencoder dimensions.
        num_latent_rows = autoencoder.dimensions['Latent']['Local'][0]
        num_latent_cols = autoencoder.dimensions['Latent']['Local'][1]
        num_texture_rows = autoencoder.dimensions['Texture']['Input'][0]
        num_texture_cols = autoencoder.dimensions['Texture']['Input'][1]
        row_expansion_ratio = autoencoder.dimensions['Texture']['Output'][0] // num_latent_rows
        col_expansion_ratio = autoencoder.dimensions['Texture']['Output'][1] // num_latent_cols

        # These sanity checks may seem obvious but you never know...
        assert num_tile_rows <= num_latent_rows, 'Tile height cannot exceed the height of the latent field.'
        assert num_tile_cols <= num_latent_cols, 'Tile width cannot exceed the width of the latent field.'
        assert num_output_rows % (row_expansion_ratio * num_tile_rows) == 0, 'Latent height inferred from the output height must be a multiple of the tile height.'
        assert num_output_cols % (col_expansion_ratio * num_tile_cols) == 0, 'Latent width inferred from the output width must be a multiple of the tile width.'

        # It is assumed that the dimensions of the input images will be accepted by the network.
        input_images = image.load(path=input_path, encoding='sRGB').unsqueeze(0)
        input_distance = utils.create_radial_distance_field(num_rows=num_texture_rows, num_cols=num_texture_cols).unsqueeze(0)
        input_batch = torch.cat([input_images, input_distance], dim=3).permute(0, 3, 1, 2)
        input_latent = autoencoder.encode(input_batch)

        # As mentioned in the assertions, the size of the shuffled latent field can be inferred from the desired output texture size.
        num_shuffled_rows = num_output_rows // row_expansion_ratio
        num_shuffled_cols = num_output_cols // col_expansion_ratio
        shuffled_latent = torch.zeros((1, input_latent.size(1), num_shuffled_rows, num_shuffled_cols), device=utils.get_device_name())

        # The shuffled latent is populated with random tiles from the input image latent.
        for row in range(0, shuffled_latent.size(2), num_tile_rows):
            for col in range(0, shuffled_latent.size(3), num_tile_cols):
                original_row_crop, original_col_crop = utils.sample_embedded_rectangle(num_outer_rows=input_latent.size(2), num_inner_rows=num_tile_rows,
                                                                                       num_outer_cols=input_latent.size(3), num_inner_cols=num_tile_cols)
                shuffled_row_crop, shuffled_col_crop = slice(row, row + num_tile_rows), slice(col, col + num_tile_cols)
                shuffled_latent[:, :, shuffled_row_crop, shuffled_col_crop] = input_latent[:, :, original_row_crop, original_col_crop]

        # The periodic latent component needs to be aligned with its relative position in the field.
        channels = {key: autoencoder.dimensions['Latent'][key][2] for key in ('Local', 'Global', 'Periodic')}
        global_field = shuffled_latent[:, channels['Local']:channels['Local'] + channels['Global'], :, :]
        shuffled_latent[:, -channels['Periodic']:, :, :] = autoencoder.derive_periodic_field(global_field)

        # The fully-convolutional nature of the SVBRDF decoder trivializes the creation of textures with arbitrary sizes.
        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.decode(shuffled_latent))
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _tile_flow(config: Configuration) -> None:
    '''
    The "tile" flow attempts to synthesize a tileable output texture from a given input texture.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, lights, viewer, camera, overlap, input_path, output_path = config.load_tile_flow()
        autoencoder.eval()

        # It is assumed that the dimensions of the input image will be accepted by the network.
        input_image = image.load(path=input_path, encoding='sRGB')
        input_distance = utils.create_radial_distance_field(num_rows=autoencoder.dimensions['Texture']['Input'][0],
                                                            num_cols=autoencoder.dimensions['Texture']['Input'][1])
        input_batch = torch.cat([input_image, input_distance], dim=2).unsqueeze(0).permute(0, 3, 1, 2)

        # As long as the perceptive field of an output pixel is less than the size of the latent field, a tileable
        # output texture can be obtained by decoding a tiling of the latent field (interpolated for good measure).
        latent_tiles_row = autoencoder.encode(input_batch).expand(3, -1, -1, -1).permute(0, 2, 3, 1)
        latent_field_row = utils.interpolate(latent_tiles_row, overlap=overlap).expand(3, -1, -1, -1)
        latent_field = utils.interpolate(latent_field_row.transpose(1, 2), overlap=overlap).transpose(0, 1).unsqueeze(0).permute(0, 3, 1, 2)

        # The center crop of the output image will be tileable as long as the latent field was smoothly convolved.
        output = autoencoder.decode(latent_field)
        output_row_padding = output.size(2) // 2 - autoencoder.dimensions['Texture']['Output'][0] // 2
        output_col_padding = output.size(3) // 2 - autoencoder.dimensions['Texture']['Output'][1] // 2
        cropped_output = output[:, :, output_row_padding:-output_row_padding, output_col_padding:-output_col_padding]

        # The fully-convolutional nature of the SVBRDF decoder trivializes the creation of textures with arbitrary sizes.
        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(cropped_output)
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _training_flow(config: Configuration) -> None:
    '''
    The "training" flow trains a neural network to map images of textures to their corresponding SVBRDF parameter maps.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    autoencoder, svbrdf, datasets, optimizer, hyperparameters = config.load_training_flow()
    training.optimize(autoencoder=autoencoder, svbrdf=svbrdf, datasets=datasets, optimizer=optimizer, **hyperparameters)


def _warp_flow(config: Configuration) -> None:
    '''
    The "warp" flow renders a plane from a source texture by sampling a local latent field uniformly at random.

    Args:
        config: Configuration specifying the parameters of the flow.
    '''
    with torch.no_grad():
        autoencoder, svbrdf, lights, viewer, camera, output_size, input_path, output_path = config.load_warp_flow()
        autoencoder.eval()

        # It is assumed that the dimensions of the input image will be accepted by the SVBRDF autoencoder network.
        input_images = image.load(path=input_path, encoding='sRGB').unsqueeze(0)
        num_texture_rows = autoencoder.dimensions['Texture']['Input'][0]
        num_texture_cols = autoencoder.dimensions['Texture']['Input'][1]
        input_distance = utils.create_radial_distance_field(num_rows=num_texture_rows, num_cols=num_texture_cols).unsqueeze(0)
        input_batch = torch.cat([input_images, input_distance], dim=3).permute(0, 3, 1, 2)

        # The expansion ratios represent the multiplicative scaling in size from the latent field to the output texture.
        row_expansion_ratio = autoencoder.dimensions['Texture']['Output'][0] // autoencoder.dimensions['Latent']['Local'][0]
        col_expansion_ratio = autoencoder.dimensions['Texture']['Output'][1] // autoencoder.dimensions['Latent']['Local'][1]

        # The value at each position in the local field is sampled uniformly at random to simulate structural noise.
        num_warped_rows = output_size[0] // row_expansion_ratio
        num_warped_cols = output_size[1] // col_expansion_ratio
        local_field = torch.rand((1, autoencoder.dimensions['Latent']['Local'][2], num_warped_rows, num_warped_cols))
        # The global field is the same everywhere to preserve the look and feel of the input texture.
        global_field = autoencoder.encoders['Global'].forward(input_batch).expand(1, num_warped_rows, num_warped_cols, -1).permute(0, 3, 1, 2)
        # The periodic field is derived directly from the global field.
        periodic_field = autoencoder.derive_periodic_field(global_field)

        # The fully-convolutional nature of the SVBRDF decoder trivializes the creation of textures with arbitrary sizes.
        latents = torch.cat([local_field, global_field, periodic_field], dim=1)
        normals, svbrdf.parameters = SVBRDFAutoencoder.interpret(autoencoder.decode(latents))
        _shade_render_save(normals=normals, svbrdf=svbrdf, lights=lights, viewer=viewer, camera=camera, path=output_path)


def _shade_render_save(normals: Tensor, svbrdf: SVBRDF, lights: List[Light], viewer: Viewer, camera: Camera, path: str) -> None:
    '''
    Shades, renders, and saves a picture to the given path from the provided normals, Lights, Viewer, Camera, and SVBRDF.

    Args:
        normals: Tensor [1, R, C, 3] of normals to use for shading.
        svbrdf: SVBRDF to use for shading.
        lights: Lights to use for shading.
        viewer: Viewer to use for shading.
        camera: Camera to use for rendering.
        path: Path to use for saving the image.
    '''
    num_rows, num_cols = normals.size(1), normals.size(2)
    num_size = max(num_rows, num_cols)
    surface = (utils.create_grid(num_rows=num_rows, num_cols=num_cols) - torch.tensor([0.5, 0.5, 0])) * torch.tensor([num_cols / num_size, num_rows / num_size, 1])
    radiance = shader.shade(surface=surface, normals=normals, lights=lights, viewer=viewer, svbrdf=svbrdf)
    picture = camera.render(surface=surface, radiance=radiance[0])
    image.save(path=path, image=picture, encoding='sRGB')
