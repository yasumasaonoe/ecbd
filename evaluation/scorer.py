"""
Computer the total perplexity/recall@10 given results.
"""

import argparse
import json
import logging
import numpy as np

from collections import defaultdict
from typing import List

from ..data_construcion.data_utils import load_jason


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


parser = argparse.ArgumentParser()
parser.add_argument("--read_from1",
                    help="Result file.",
                    default="")
parser.add_argument("--read_from2",
                    help="Result file.",
                    default="")
parser.add_argument("--paths",
                    help="List of result files.",
                    nargs='+',
                    type=str)
parser.add_argument("--save_to",
                    help="A JSON file that sentences will be saved.",
                    default="")
parser.add_argument("--mode",
                    help="Masking mode.",
                    choices=['perplexity', 'recall'])


def get_perplexity(
        results: List,
        save_to: str,
        ignore_token: str = '<extra_id_1>'
):
    out_dict = {}

    # Micro
    perplexity = defaultdict(list)
    for r in results:
        # Drop '<extra_id_1>'
        if r['loss_per_token'][-1][0] == ignore_token:
            r['loss_per_token'] = r['loss_per_token'][:-1]
        span_type = r['span_type']
        perplexity[span_type].append(
            #np.mean([l[1] for l in r['nll_per_token']]))
            np.mean([l[1] for l in r['loss_per_token']]))

    entities = defaultdict(list)
    # Macro
    for r in results:
        # Drop '<extra_id_1>'
        if r['loss_per_token'][-1][0] == ignore_token:
            r['loss_per_token'] = r['loss_per_token'][:-1]
        ex_id = r['ex_id'].split('_')
        pageid = ex_id[-3]
        #entities[pageid].append(np.meLan([l[1] for l in r['nll_per_token']]))
        entities[pageid].append(np.mean([l[1] for l in r['loss_per_token']]))
        #entities[pageid].append(np.sum([l[1] for l in r['loss_per_token']]))

    out_dict['perplexity_micro'] = np.exp(
        np.mean(
            [s for lst in perplexity.values() for s in lst]
        )
    )

    out_dict['perplexity_macro'] = np.exp(
        np.mean(
            [np.mean(lst) for lst in entities.values()]
        )
    )

    for span_type, lst in perplexity.items():
        out_dict[f'n_{span_type}'] = len(lst)
        out_dict[f'perplexity_{span_type}'] = np.exp(np.mean(lst))


    with open(save_to, 'w') as f:
        json.dump(out_dict, f)

    print(out_dict)


def get_recall(
        results: List,
        save_to: str
):
    out_dict = {'n_examples': len(results)}

    # Micro
    recall = defaultdict(list)
    for r in results:
        span_type = r['span_type']
        gold_str = r['gold_span']['<extra_id_0>']
        pred_strs = [p['<extra_id_0>'] for p in r['pred_spans']]
        em = 1 if gold_str in pred_strs else 0
        recall[span_type].append(em)

    entities = defaultdict(list)
    # Macro
    for r in results:
        ex_id = r['ex_id'].split('_')
        pageid = ex_id[-3]
        gold_str = r['gold_span']['<extra_id_0>']
        pred_strs = [p['<extra_id_0>'] for p in r['pred_spans']]
        em = 1 if gold_str in pred_strs else 0
        entities[pageid].append(em)

    out_dict['recall_micro'] = np.mean(
            [s for lst in recall.values() for s in lst]
    )

    out_dict['recall_macro'] = np.mean(
            [np.mean(lst) for lst in entities.values()]
    )

    for span_type, lst in recall.items():
        out_dict[f'n_{span_type}'] = len(lst)
        out_dict[f'recall_{span_type}'] = np.mean(lst)


    with open(save_to, 'w') as f:
        json.dump(out_dict, f)

    print(out_dict)



def main():
    args = parser.parse_args()

    results = []
    if args.read_from1 != '' or args.read_from2 != '':
        if args.read_from1 != '':
            logger.info(f'Load data from {args.read_from1}')
            results += load_jason(args.read_from1)
            logger.info('len(results) = {}'.format(len(results)))

        if args.read_from2 != '':
            logger.info(f'Load data from {args.read_from2}')
            results += load_jason(args.read_from2)
            logger.info('len(results) = {}'.format(len(results)))
    elif args.paths:
        for path in args.paths:
            logger.info(f'Load data from {path}')
            results += load_jason(path)
            logger.info('len(results) = {}'.format(len(results)))
    else:
        raise ValueError

    suffix = 'results.perplexity' if args.mode == 'perplexity' else 'results.recall'
    if args.save_to == "":
        save_to = args.save_to + suffix
    else:
        save_to = args.save_to

    if args.mode == 'perplexity':
        get_perplexity(results, save_to)
    else:  # Recall
        get_recall(results, save_to)

    logger.info(f'Saved to {args.save_to}')


if __name__ == '__main__':
    main()
