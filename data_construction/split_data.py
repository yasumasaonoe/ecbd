"""
Split data
"""

import argparse
import logging
import random

from data_utils import split_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


parser = argparse.ArgumentParser()
parser.add_argument("--read_from",
                    help="Sentences.",
                    default="")
parser.add_argument("--save_to",
                    help="A JSON file that sentences will be saved.",
                    default="")
parser.add_argument("--entity_file_path",
                    help="A JSON file that contains entity info.",
                    default=None)
parser.add_argument("--definition_file",
                    help="A JSON file that contains definitions.",
                    default="")
parser.add_argument("--span_file",
                    help="A JSON file that contains span info.",
                    default="")
parser.add_argument("--distance",
                    help="Batch size.",
                    default=10,
                    type=int)
parser.add_argument("--min_n_sentences",
                    help="Minimum number of sentences.",
                    default=10,
                    type=int)
parser.add_argument("--max_n_sentences",
                    help="Maximum number of sentences.",
                    default=500,
                    type=int)

SEED = 2022

def main():
    args = parser.parse_args()
    logger.info(
        f'Load data from {args.read_from}'
    )

    random.seed(SEED)
    rng = random

    split_dataset(args.read_from,
                  args.save_to,
                  args.definition_file,
                  args.span_file,
                  args.entity_file_path,
                  distance=args.distance,
                  min_n_sentences=args.min_n_sentences,
                  max_n_sentences=args.max_n_sentences,
                  rng=rng)

    logger.info(
        f'Saved to {args.save_to + "_*"}'
    )


if __name__ == '__main__':
    main()
