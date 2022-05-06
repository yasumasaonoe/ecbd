"""
Fill entity spans with pronouns or a specified token.
"""

import argparse
import logging
import torch

from transformers import T5ForConditionalGeneration,T5Tokenizer

from ..evaluation.eval_utils import predict_pronouns_for_entities, replace_entities_with_token
from .data_utils import load_jason


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
parser.add_argument("--mode",
                    help="Masking mode.",
                    choices=['pronoun', 'token'])
parser.add_argument("--checkpoint",
                    help="Masking mode.",
                    default='t5-large')
parser.add_argument("--batch_size",
                    help="Batch size.",
                    default=1,
                    type=int)


def main():
    args = parser.parse_args()
    logger.info(f'args: {args}')
    logger.info(
        f'Load data from {args.read_from}'
    )

    sentences = load_jason(args.read_from)

    logger.info(f'Use {args.checkpoint}')

    if args.mode == 'pronoun':
        logger.info(f'Use {args.checkpoint}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
        tokenizer = T5Tokenizer.from_pretrained(args.checkpoint)
        model = model.to(device)
        predict_pronouns_for_entities(
            sentences,
            args.save_to,
            model,
            tokenizer,
            device,
            masked_token='<extra_id_0>',
            batch_size=args.batch_size
        )
    elif args.mode == 'token':
        replace_entities_with_token(
            sentences,
            args.save_to,
            'the entity',
            masked_token='<extra_id_0>'
        )
    else:
        raise NotImplementedError

    logger.info(
        f'Saved to {args.save_to}'
    )


if __name__ == '__main__':
    main()
