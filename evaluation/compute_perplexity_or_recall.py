"""
Compute NLL on the masked spans and save the results.
"""

import argparse
import logging
import random
import torch

from ..evaluation.eval_utils import compute_perplexity_and_recall
from ..data_construction.data_utils import load_jason, \
    format_example_for_bart, format_example_for_gpt


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
parser.add_argument("--checkpoint",
                    help="Masking mode.",
                    default='t5-large')
parser.add_argument("--batch_size",
                    help="Batch size.",
                    default=1,
                    type=int)
parser.add_argument("--k",
                    help="Top k.",
                    default=1,
                    type=int)
parser.add_argument("--max_sentence_len",
                    help="Max sentence length.",
                    default=1,
                    type=int)
parser.add_argument("--beam_size",
                    help="Beam size.",
                    default=1,
                    type=int)
parser.add_argument("--mode",
                    help="Perplexity or recall.",
                    default='perplexity',
                    type=str)
parser.add_argument("--replace_token",
                    help="Replace token.",
                    default=None,
                    type=str)
parser.add_argument("--n_definition",
                    help="Number of definiton sentences to be added.",
                    default=0,
                    type=int)
parser.add_argument("--use_random_definition",
                    help="Use random definition.",
                    type=str,
                    default="False")

SEED = 2022

def main():
    args = parser.parse_args()
    logger.info(f'Load data from {args.read_from}')
    args.use_random_definition = True if args.use_random_definition == 'True'\
        else False

    print(args)

    examples = load_jason(args.read_from)

    logger.info('len(sentences1) = {}'.format(len(examples)))
    logger.info(f'Use {args.checkpoint}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if 't5' in args.checkpoint:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
        tokenizer = T5Tokenizer.from_pretrained(args.checkpoint)
    elif 'bart' in args.checkpoint:
        from transformers import BartForConditionalGeneration, BartTokenizer
        model = BartForConditionalGeneration.from_pretrained(
            f"facebook/{args.checkpoint}", forced_bos_token_id=0)
        tokenizer = BartTokenizer.from_pretrained(f"facebook/{args.checkpoint}")
        examples = [format_example_for_bart(ex) for ex in examples]
    elif 'gpt' in args.checkpoint:
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        model = GPTNeoForCausalLM.from_pretrained(
            f"EleutherAI/{args.checkpoint}").to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/{args.checkpoint}")
        tokenizer.pad_token = tokenizer.eos_token
        examples = [format_example_for_gpt(ex, tokenizer.pad_token) for ex in examples]
    else:
        raise ValueError(f'Invalid checkpoint: {args.checkpoint}')

    model = model.to(device)

    if args.use_random_definition:
        random.seed(SEED)
        rng = random
    else:
        rng = None

    compute_perplexity_and_recall(
        examples,
        args.save_to,
        model,
        tokenizer,
        device,
        batch_size=args.batch_size,
        k=args.k,
        max_sentence_len=args.max_sentence_len,
        beam_size=args.beam_size,
        mode=args.mode,
        n_definition=args.n_definition,
        use_random_definition=args.use_random_definition,
        replace_token=args.replace_token,
        rng=rng
    )

    logger.info(
        f'Saved to {args.save_to}'
    )


if __name__ == '__main__':
    main()