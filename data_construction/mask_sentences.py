"""
Mask sentences extracted by 'select_wikipedia_sentences.py'

"""

import argparse
import logging
import numpy as np

from data_utils import format_input_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


parser = argparse.ArgumentParser()
parser.add_argument("--read_from",
                    help="Entity data.",
                    default="")
parser.add_argument("--save_to",
                    help="A JSON file that sentences will be saved.",
                    default="")
parser.add_argument("--masking_mode",
                    help="Masking mode.",
                    choices=['span', 'span_w_fake_entity', 'entity'])
parser.add_argument("--mask_type",
                    help="Mask type.",
                    choices=['noun_phrases', 'random'])
parser.add_argument("--mask_after_entity",
                    help="Always mask after entity mentions.",
                    default=True)


SEED = 2022

def main():
    args = parser.parse_args()
    logger.info(
        f'Load data from {args.read_from}'
    )

    logger.info(f'Generate input data with this masking mode:'
                f' {args.masking_mode}')
    logger.info(f'Mask Type:'
                f' {args.mask_type}')

    if args.mask_type == 'random':
        # Seed the generator.
        np.random.seed(SEED)
        rng = np.random
    else:
        rng = None

    format_input_data(args.read_from,
                      args.save_to,
                      masking_mode=args.masking_mode,
                      mask_type=args.mask_type,
                      mask_after_entity=args.mask_after_entity,
                      rng=rng)
    logger.info(
        f'Saved to {args.save_to}'
    )


if __name__ == '__main__':
    main()
