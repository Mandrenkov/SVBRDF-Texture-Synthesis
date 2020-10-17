import argparse
import flow
import logging
import torch

from config import Configuration


if __name__ == '__main__':
    # Parse the command-line arguments and display some help text too.
    parser = argparse.ArgumentParser(description='The "Procedural Texture Synthesis with Machine Learning" research project.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flow', type=str, required=True, choices=['album', 'blend', 'extract', 'feedback', 'merge',
                        'merl', 'morph', 'mosaic', 'relight', 'render', 'shuffle', 'tile', 'training', 'warp'], metavar='FLOW',
                        help='"album" renders an image composed of blended textures; '
                           + '"blend" statically interpolates two textures; '
                           + '"extract" extracts a sample from a Dataset; '
                           + '"feedback" simulates a texture feedback loop; '
                           + '"merge" smoothly interpolates two overlapping textures; '
                           + '"merl" fits an SVBRDF to the MERL 100 dataset; '
                           + '"morph" discretely interpolates two disjoint textures; '
                           + '"mosaic" reconstructs a scaled image with arbitrary lighting; '
                           + '"relight" reconstructs an image with arbitrary lighting; '
                           + '"render" renders a texture from a set of parameter maps; '
                           + '"shuffle" expands an image by shuffling latent tiles; '
                           + '"tile" synthesizes a tileable version of a texture; '
                           + '"training" trains an SVBRDF autoencoder network; '
                           + '"warp" expands an image by sampling a random local latent field')
    parser.add_argument('--config', type=str, help='path to the YAML configuration file')
    parser.add_argument('--seed', type=int, default=1, help='seed for random number generation')
    parser.add_argument('--verbosity', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                                       metavar='LEVEL', help='logging verbosity')
    parser.add_argument('--cuda', action='store_true', help='enable CUDA acceleration')
    args = parser.parse_args()

    # Initialize any global state, including RNG seeds and logging preferences.
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s.', datefmt='%Y-%m-%d %H:%M:%S',
                        level=args.verbosity)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.enabled = True  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
    else:
        torch.backends.cudnn.enabled = False  # type: ignore
        torch.set_default_tensor_type(torch.FloatTensor)

    # The configuration files are named in a predictable way for a reason.
    config_path = args.config if args.config is not None else f'configs/{args.flow}.yaml'

    # Load the Configuration file and run the corresponding flow.
    config = Configuration(config_path)
    flow.execute(args.flow, config)
