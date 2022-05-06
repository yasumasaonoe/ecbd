"""
Evaluate the perplexity delta (drop) between two sentences (e.g., one with
an entity mention and another with a pronoun). This script is used for
quality check only.
"""


import argparse
import logging
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer

from ..evaluation.eval_utils import evaluate_sentences
from ..data_construction.data_utils import load_jason


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


parser = argparse.ArgumentParser()
parser.add_argument("--read_from1",
                    help="Entity data.",
                    default="")
parser.add_argument("--read_from2",
                    help="Entity data.",
                    default="")
parser.add_argument("--save_to",
                    help="A JSON file that sentences will be saved.",
                    default="")
parser.add_argument("--checkpoint",
                    help="Masking mode.",
                    default='t5-large')
parser.add_argument("--do_srl_filtering",
                    help="Do srl filtering.",
                    default=False,
                    type=bool)
parser.add_argument("--batch_size",
                    help="Batch size.",
                    default=1,
                    type=int)


def main():
    args = parser.parse_args()
    logger.info(f'Load data from {args.read_from1}')
    logger.info(f'Load data from {args.read_from2}')

    sentences1 = load_jason(args.read_from1)
    sentences2 = load_jason(args.read_from2)

    logger.info('len(sentences1) = {}'.format(len(sentences1)))
    logger.info('len(sentences2) = {}'.format(len(sentences2)))

    logger.info(f'Use {args.checkpoint}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint)
    model = model.to(device)

    evaluate_sentences(
        sentences1,
        sentences2,
        args.save_to,
        model,
        tokenizer,
        device,
        min_sentence_len=5,
        batch_size=args.batch_size,
        max_examples=100000000,  # all examples
        do_srl_filtering=args.do_srl_filtering
    )

    logger.info(
        f'Saved to {args.save_to}'
    )


if __name__ == '__main__':
    main()
