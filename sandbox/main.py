import argparse
import random

from synthesizers import CoherentPixelNeighbourhoodSynthesizer, RasterPixelNeighbourhoodSynthesizer, SpiralPixelNeighbourhoodSynthesizer
from typing import cast, Tuple

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to the "Procedural Texture Synthesis with Machine Learning" sandbox!', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flow', type=str, required=True, choices=['cpns', 'rpns', 'spns'], help='flow to execute.')
    parser.add_argument('--seed', type=int, default=1, help='seed for the random number generators.')
    # Source image parameters.
    parser.add_argument('--source_image', '-si', type=str, default='textures/pebbles.png', help='path to the source image.')
    parser.add_argument('--source_image_size', '-sis', type=int, nargs=2, default=[64, 64], help='width and height of the source image.')
    # Output image parameters.
    parser.add_argument('--output_image', '-oi', type=str, default='results/output.png', help='path to the output image.')
    parser.add_argument('--output_image_size', '-ois', type=int, nargs=2, default=[64, 64], help='width and height of the output image.')
    # TextureSynthesizer parameters.
    parser.add_argument('--neighbourhood_padding', '-np', type=int, nargs='+', default=[3], help='padding applied to a pixel to form the neighbourhood window of each layer in an image pyramid.')
    parser.add_argument('--seed_padding', '-sp', type=int, default=1, help='padding applied to a pixel to form the seed window.')
    parser.add_argument('--coherent_set_size', '-css', type=int, default=2, help='number of nearest neighbours to consider in the coherence synthesizer.')
    parser.add_argument('--tsvq_branching_factor', '-tsvq', type=int, default=0, help='branching factor of the TSVQ tree in the raster synthesizer.')

    args = parser.parse_args()

    random.seed(args.seed)

    # Cast the source and output images sizes into tuple pairs.
    args.source_image_size = cast(Tuple[int, int], tuple(args.source_image_size))
    args.output_image_size = cast(Tuple[int, int], tuple(args.output_image_size))

    if args.flow == 'cpns':
        CoherentPixelNeighbourhoodSynthesizer(source_image_path=args.source_image,
                                              source_image_size=args.source_image_size,
                                              output_image_path=args.output_image,
                                              output_image_size=args.output_image_size,
                                              coherent_set_size=args.coherent_set_size).synthesize()
    elif args.flow == 'rpns':
        RasterPixelNeighbourhoodSynthesizer(source_image_path=args.source_image,
                                            source_image_size=args.source_image_size,
                                            output_image_path=args.output_image,
                                            output_image_size=args.output_image_size,
                                            neighbourhood_padding=args.neighbourhood_padding,
                                            tsvq_branching_factor=args.tsvq_branching_factor).synthesize()
    elif args.flow == 'spns':
        SpiralPixelNeighbourhoodSynthesizer(source_image_path=args.source_image,
                                            source_image_size=args.source_image_size,
                                            output_image_path=args.output_image,
                                            output_image_size=args.output_image_size,
                                            seed_padding=args.seed_padding,
                                            neighbourhood_padding=args.neighbourhood_padding).synthesize()
