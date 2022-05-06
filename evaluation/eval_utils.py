"""
Utils for evaluation
"""

import json
import numpy as np
import re
import torch

from allennlp.predictors import Predictor
from torch import device
from torch.nn import Module, CrossEntropyLoss
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from typing import Any, Dict, List, Tuple

from ..data_construction.data_utils import clean_sentence, save_json


PRONOUNS = [
    'I', 'me', 'my', 'Me',
    'we', 'us', 'We', 'Us',
    'you', 'You',
    'she', 'her', 'She', 'Her',
    'he', 'him', 'He', 'Him',
    'they', 'them', 'They',
    'it', 'It',
    'that', 'those', 'That', 'Those',
    'this', 'these', 'This', 'These',
    'someone', 'something', 'Someone', 'Something'
]


def compute_perplexity(
        masked_sentences: List[str],
        answers: List[str],
        model: Module,
        tokenizer: PreTrainedTokenizerFast,
        device: device,
        left_context: List[str] = None,
        right_context: List[str] = None
):
    assert len(masked_sentences) == len(answers)
    batch_size = len(masked_sentences)

    input_encoding = tokenizer(masked_sentences,
                               return_tensors='pt',
                               padding=True,
                               truncation=True)
    input_ids = input_encoding["input_ids"].to(device)
    attention_mask = input_encoding["attention_mask"].to(device)
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    if 'gpt' in model.name_or_path:
        label_encoding = tokenizer(answers,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True)
        label_ids = label_encoding["input_ids"].to(device)
        label_attention_mask = label_encoding["attention_mask"].to(device)
        left_context_encoding = tokenizer(left_context,
                                          return_tensors='pt',
                                          padding=True,
                                          truncation=True)

        right_context_encoding = tokenizer(right_context,
                                           return_tensors='pt',
                                           padding=True,
                                           truncation=True)

        left_context_ids = left_context_encoding["input_ids"].to(device)
        right_context_ids = right_context_encoding["input_ids"].to(device)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
    else:  # T5, BART
        label_encoding = tokenizer(answers,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True)
        label_ids = label_encoding["input_ids"].to(device)
        label_attention_mask = label_encoding["attention_mask"].to(device)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=label_attention_mask,
            labels=label_ids
        )
        loss = loss_fct(output.logits.view(-1, output.logits.size(-1)), label_ids.view(-1))
    perp_loss = []
    for i, l in enumerate(loss.view(batch_size, -1)):
        if 't5' in model.name_or_path:
            # Remove </s>, <pad>
            n_tokens = label_attention_mask[i].sum() - 2
            # Exclude <extra_id_0>, <extra_id_1>
            perplexity = torch.exp(
                (l * label_attention_mask[i])[1:n_tokens].mean()).item()
            loss_per_token = list(
                zip(tokenizer.convert_ids_to_tokens(
                    label_ids[i].cpu().detach().numpy().tolist())[1:n_tokens],
                    [float(s) for s in l.cpu().detach().numpy()[1:n_tokens]]
                    )
            )
        elif 'bart' in model.name_or_path:
            masked_sentence_ids = input_ids[i]
            answer_ids = label_ids[i]
            start_loc = (masked_sentence_ids == tokenizer.mask_token_id).nonzero(
                as_tuple=True)[0].item()
            assert masked_sentence_ids.size(0) <= answer_ids.size(0)
            end_loc = start_loc + ((answer_ids != 1).sum().item() -
                                   (masked_sentence_ids != 1).sum().item() + 1)
            perplexity = torch.exp(
                (l * label_attention_mask[i])[start_loc:end_loc].mean()).item()
            loss_per_token = list(
                zip(tokenizer.convert_ids_to_tokens(
                    answer_ids.cpu().detach().numpy().tolist())[start_loc:end_loc],
                    [float(s) for s in l.cpu().detach().numpy()[start_loc:end_loc]]
                    )
            )

            # print(tokenizer.convert_ids_to_tokens(
            #     masked_sentence_ids.cpu().detach().numpy().tolist())[:])
            # print(tokenizer.convert_ids_to_tokens(
            #     masked_sentence_ids.cpu().detach().numpy().tolist())[
            #       start_loc:end_loc])
            # print(tokenizer.convert_ids_to_tokens(
            #         answer_ids.cpu().detach().numpy().tolist())[start_loc:end_loc])
            # print()


        elif 'gpt' in model.name_or_path:
            masked_sentence_ids = input_ids[i]
            total_len = (masked_sentence_ids != tokenizer.eos_token_id).sum().item()
            left_len = (left_context_ids[i] != tokenizer.eos_token_id).sum().item()
            right_len = (right_context_ids[i] != tokenizer.eos_token_id).sum().item()
            start_loc = left_len
            span_len = total_len - left_len - right_len
            end_loc = start_loc + span_len

            perplexity = torch.exp(
                l[start_loc-1:end_loc-1].mean()).item()
            # print(tokenizer.convert_ids_to_tokens(
            #     masked_sentence_ids.cpu().detach().numpy().tolist())[:])
            # print(tokenizer.convert_ids_to_tokens(
            #         masked_sentence_ids.cpu().detach().numpy().tolist())[
            #         start_loc:-end_loc])
            # print([float(s) for s in l.cpu().detach().numpy()[:]])
            # print([float(s) for s in l.cpu().detach().numpy()[
            #                            start_loc - 1:-end_loc]])
            # print(start_loc, -end_loc)
            # print(l.size(), masked_sentence_ids.size())
            loss_per_token = list(
                zip(tokenizer.convert_ids_to_tokens(
                    masked_sentence_ids.cpu().detach().numpy().tolist())[
                    start_loc:end_loc],
                    [float(s) for s in l.cpu().detach().numpy()[
                                       start_loc-1:end_loc-1]]
                    )
            )
            if len(loss_per_token) == 0:
                print(tokenizer.convert_ids_to_tokens(
                    masked_sentence_ids.cpu().detach().numpy().tolist())[:])
                print(tokenizer.convert_ids_to_tokens(
                        masked_sentence_ids.cpu().detach().numpy().tolist())[
                        start_loc:end_loc])
                print([float(s) for s in l.cpu().detach().numpy()[:]])
                print([float(s) for s in l.cpu().detach().numpy()[
                                           start_loc - 1:end_loc - 1]])
                print(start_loc, end_loc)
                print(l.size(), masked_sentence_ids.size())
                print()
                raise

        else:
            raise ValueError(f"{model.name_or_path} is not supported.")
        perp_loss.append((perplexity, loss_per_token))

    return perp_loss


def predict_pronouns_for_entities(
    sentences: List[Dict],
    save_to: str,
    model: Module,
    tokenizer: PreTrainedTokenizerFast,
    device: device,
    masked_token: str = '<extra_id_0>',
    batch_size: int = 1
):
    pronoun_ids = list(
        set(
            [i for ids in tokenizer(PRONOUNS)['input_ids'] for i in ids if
              i not in [tokenizer.eos_token_id, 3]]  # 3 == '_' for T5 tokenizer
        )
    )
    pronoun_ids = torch.tensor(pronoun_ids)

    # Create batches
    n_batches = len(sentences) // batch_size
    if len(sentences) % batch_size != 0:
        n_batches += 1
    sentences = [sentences[i * batch_size:(i + 1) * batch_size]
                 for i in range(n_batches)]

    with open(save_to, 'w') as f:

        for batch_id, batch in tqdm(enumerate(sentences)):
            encoding = tokenizer([sent['masked_sentence'] for sent in batch],
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=True)
            input_ids = encoding["input_ids"].to(device)
            attention_masks = encoding["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=3,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True
            )
            for i in range(len(batch)):
                # Index=1 is the first word predicted.
                sentence = batch[i]
                scores = outputs.scores[1][i]  # vocab size
                score_mask = torch.ones_like(scores) * -np.inf
                score_mask[pronoun_ids] = 0.
                scores += score_mask
                predicted_pronoun = torch.argmax(scores).item()
                predicted_pronoun = tokenizer.convert_ids_to_tokens(
                    predicted_pronoun)
                predicted_pronoun = tokenizer.convert_tokens_to_string(
                    predicted_pronoun)
                assert masked_token in sentence['masked_sentence'], sentence
                split_sentence = sentence['masked_sentence'].split(masked_token)
                if len(split_sentence) != 2:
                    print('WARNING: bad sentence={}'.format(sentence['ex_id']))
                    continue
                left, right = split_sentence
                # Hacky way to handle cap
                if sentence['mask_loc'][0] == 0:
                    predicted_pronoun = predicted_pronoun.capitalize()
                elif predicted_pronoun != 'I':
                    predicted_pronoun = predicted_pronoun.lower()
                filled_sentence = left + predicted_pronoun + right
                sentence['filled_sentence'] = filled_sentence
                sentence['replace_token'] = predicted_pronoun
                json.dump(sentence, f)
                f.write('\n')


def replace_entities_with_token(
    sentences: List[Dict],
    save_to: str,
    replace_token: str,
    masked_token: str = '<extra_id_0>'
):

    with open(save_to, 'w') as f:

        for sent_id, sentence in tqdm(enumerate(sentences)):
            assert masked_token in sentence['masked_sentence']
            split_sentence = sentence['masked_sentence'].split(masked_token)
            if len(split_sentence) != 2:
                print('WARNING: bad sentence={}'.format(sentence['ex_id']))
                continue
            left, right = split_sentence
            if sentence['mask_loc'][0] == 0:
                filled_sentence = left + replace_token.capitalize() + right
                sentence['replace_token'] = replace_token.capitalize()
            else:
                filled_sentence = left + replace_token + right
                sentence['replace_token'] = replace_token
            sentence['filled_sentence'] = filled_sentence

            json.dump(sentence, f)
            f.write('\n')


def replace_span(
        sentence: str,
        token: str,
        span_loc_char: Tuple[int, int]) -> str:
    span_start, span_end = span_loc_char
    return sentence[:span_start] + token + sentence[span_end:]


def predict_topk(
    masked_sentences: List[str],
    answers: List[str],
    model: Module,
    tokenizer: PreTrainedTokenizerFast,
    device: device,
    k: int = 10,
    max_sentence_len: int = 20,
    beam_size: int = 10,
    left_context: List[str] = None,
    right_context: List[str] = None
):
    batch_size = len(masked_sentences)
    input_encoding = tokenizer(masked_sentences,
                               return_tensors='pt',
                               padding=True,
                               truncation=True)
    input_ids = input_encoding["input_ids"].to(device)
    attention_mask = input_encoding["attention_mask"].to(device)

    label_encoding = tokenizer(answers,
                               return_tensors='pt',
                               padding=True,
                               truncation=True)
    label_ids = label_encoding["input_ids"].to(device)

    if 't5' in model.name_or_path:
        beam_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_sentence_len,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )

    else:
        left_context_encoding = tokenizer(left_context,
                                          return_tensors='pt',
                                          padding=True,
                                          truncation=True)
        try:
            right_context_encoding = tokenizer(right_context,
                                               return_tensors='pt',
                                               padding=True,
                                               truncation=True)
        except:
            print(masked_sentences)
            print(right_context)
            raise
        left_context_ids = left_context_encoding["input_ids"].to(device)
        left_context_attention_mask = left_context_encoding["attention_mask"].to(device)
        right_context_ids = right_context_encoding["input_ids"].to(device)
        if 'bart' in model.name_or_path:
            pass
        elif 'gpt' in model.name_or_path:
            beam_outputs = model.generate(
                input_ids=left_context_ids,
                attention_mask=left_context_attention_mask,
                max_length=max_sentence_len,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        else:
            raise ValueError(f"{model.name_or_path} is not supported.")


    output_sequences = beam_outputs['sequences'].view(batch_size, beam_size, -1)
    results = [[seq for seq in ex] for ex in output_sequences]

    all_unique_results = []
    all_seq_ids = []
    for i, ex in enumerate(results):
        unique_results = []
        seq_ids = []
        for j, seq in enumerate(ex):
            if 't5' in model.name_or_path:
                seq = tokenizer.decode(seq)
                ans = extract_answers_t5(seq, 1)
            elif 'bart' in model.name_or_path:
                ans = extract_answers_bart(
                    seq,
                    input_ids[i],
                    label_ids[i],
                    tokenizer
                )
            else:
                raise ValueError(f"{model.name_or_path} is not supported.")
            if len(ans) == 1 and ans not in unique_results:
                unique_results.append(ans)
                seq_ids.append(j)
                if len(unique_results) == k:
                    all_unique_results.append(unique_results)
                    all_seq_ids.append(all_seq_ids)
                    break
        else:
            all_unique_results.append(unique_results)
            all_seq_ids.append(all_seq_ids)

    return all_unique_results, beam_outputs, all_seq_ids


def extract_answers_t5(seq, n_blank):
    # TODO: use indices instead of string
    #seq = seq.lstrip('<pad>').strip()
    pattern = r'>(.*?)<'
    answers = re.findall(pattern, seq)
    answers = [a for a in answers if a.strip()]
    answers_d = {}
    i = 0
    for ans in answers:
        if ans.strip():
            answers_d[f'<extra_id_{i}>'] = ans.strip()
            i += 1
            if i == n_blank:
                break
    #if not answers and '<extra_id_0>' in seq:
    #    answers_d['<extra_id_0>'] = seq.split('<extra_id_0>')[-1].strip()
    return answers_d


def extract_answers_bart(
        seq,
        input_ids,
        label_ids,
        tokenizer
):
    """Support 1 mask only."""
    start = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
    assert input_ids.size(0) <= label_ids.size(0)
    end = start + ((label_ids != 1).sum().item() - (input_ids != 1).sum().item() + 1)

    # Find the left and right sequences.
    input_left_seq = input_ids[:start]
    input_right_seq = input_ids[start + 1:][input_ids[start + 1:] !=
                                            tokenizer.pad_token_id]  # drop pads

    # Generated seq always begins with <\s>, id=2.
    if seq[0] == tokenizer.eos_token_id:
        output_left_seq = seq[1:start + 1]
        start += 1
        end += 1
    elif seq[0] == tokenizer.bos_token_id:
        # probably this shouldn't happen...
        output_left_seq = seq[:start + 1]
    else:
        # probably this shouldn't happen...
        raise

    output_right_seq = seq[seq != tokenizer.pad_token_id][
                       -input_right_seq.size(0):]
    # print(start, input_ids.size())
    # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
    #     seq[start:end])))
    # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
    #     input_ids)))
    # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
    #     seq)))
    # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
    #     input_left_seq)))
    # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
    #     output_left_seq)))
    # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
    #     input_right_seq)))
    # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
    #     output_right_seq)))
    #
    # print()

    # Check if the left and right sequences match.
    # assert input_left_seq.size(0) == output_left_seq.size(0)
    # assert (input_left_seq == output_left_seq).all().item()
    # assert input_right_seq.size(0) == output_right_seq.size(0)
    # assert (input_right_seq == output_right_seq).all().item()

    # Extract predicted tokens
    #ans_ids = seq[seq != tokenizer.pad_token_id][output_left_seq.size(0):output_right_seq.size(0)]
    ans_ids = seq[start:end]
    ans_str = tokenizer.decode(ans_ids, skip_special_tokens=True)
    # print(f'ans_str:{ans_str}')
    answers_d = {f'<extra_id_0>': ans_str.strip()}
    return answers_d


def compute_topk_accuracy_em(pred, gold, k=1):
    assert len(pred) == len(gold)
    n_spans = 0
    n_correct = 0
    for p, g in zip(pred, gold):
        for span_id, gold_str in g.items():
            pred_strs = [_p[span_id] for _p in p[:k]]
            if gold_str in pred_strs:
                n_correct += 1
            n_spans += 1
    return n_correct / float(n_spans) if n_correct else 0.


def evaluate_sentences(sentences1: List[Dict],
                       sentences2: List[Dict],
                       save_to: str,
                       model: Module,
                       tokenizer: PreTrainedTokenizerFast,
                       device: device,
                       batch_size: int = 1,
                       min_sentence_len: int = 5,
                       max_examples: int = 100000000,
                       do_srl_filtering: bool = False):

    good = 0
    total = 0

    # Create batches
    n_batches = len(sentences1) // batch_size
    if len(sentences1) % batch_size != 0:
        n_batches += 1
    sentences1 = [sentences1[i * batch_size:(i + 1) * batch_size]
                 for i in range(n_batches)]

    # Convert a list to a dictionary
    sentences2_d = {s['ex_id']: s for s in sentences2}

    with open(save_to, 'w') as f:

        for batch_id, batch in tqdm(enumerate(sentences1)):
            masked_sentences1 = []
            masked_sentences2 = []
            target_sequences = []
            srl_filter_results = []
            for sentence1 in batch:
                ex_id = sentence1['ex_id'].split('_')
                ex_id[-1] = '0'
                ex_id = '_'.join(ex_id)
                sentence2 = sentences2_d[ex_id]
                replace_token = sentence2['replace_token']

                srl_filter_pass = False
                if do_srl_filtering:
                    model_path = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
                    srl_model = Predictor.from_path(model_path)
                    srl_outputs = srl_model.predict(sentence1['sentence'])
                    srl_words = srl_outputs['words']
                    srl_frames = srl_outputs['verbs']
                    srl = []
                    for frame in srl_frames:
                        srl_tags = frame['tags']
                        assert len(srl_tags) == len(srl_words)
                        srl_spans = {t.lstrip('B-').lstrip('I-'): [] for t in
                                     set(srl_tags) if t != 'O'}
                        for t, w in zip(srl_tags, srl_words):
                            _t = t.lstrip('B-').lstrip('I-')
                            if _t in srl_spans:
                                srl_spans[_t].append(w)
                        srl_spans = {t: ' '.join(s) for t, s in srl_spans.items()}
                        srl_sbj = ''
                        srl_obj = ''
                        srl_pred = ''
                        if 'V' in srl_spans:
                            srl_pred = srl_spans['V']
                        if 'ARG0' in srl_spans:
                            srl_sbj = srl_spans['ARG0']
                            if 'ARG1' in srl_spans:
                                srl_obj = srl_spans['ARG1']
                        elif 'ARG1' in srl_spans:
                            srl_sbj = srl_spans['ARG1']
                            if 'ARG2' in srl_spans:
                                srl_obj = srl_spans['ARG2']
                        srl.append({'sbj': srl_sbj,
                                    'obj': srl_obj,
                                    'pred': srl_pred})

                    for _s in srl:
                        if sentence1['entity_span_str'] in _s['sbj'] and \
                                sentence1['mask_span_str'] in _s['obj']:
                            srl_filter_pass = True
                    srl_filter_results.append(srl_filter_pass)

                # Sanity check
                assert sentence1['original_sentence'] == sentence2[
                    'original_sentence'], (sentence1, sentence2)
                assert clean_sentence(
                    replace_span(
                        sentence1['original_sentence'],
                        sentence1['mask_token'],
                        sentence1['masked_span_loc_char']
                    )
                ) == sentence1['masked_sentence'], sentence1
                assert clean_sentence(
                    replace_span(
                        sentence2['original_sentence'],
                        sentence2['mask_token'],
                        sentence2['masked_span_loc_char']
                    )
                ) == sentence2['masked_sentence'], sentence2
                assert sentence1['entity_span_str'] == sentence2[
                    'entity_span_str']

                masked_sentence1 = sentence1['masked_sentence']
                original_sentence = sentence2['original_sentence']
                span_mask_start, span_mask_end = sentence1[
                    'masked_span_loc_char']
                entity_mask_start, entity_mask_end = sentence2[
                    'masked_span_loc_char']
                if span_mask_end < entity_mask_start:
                    masked_sentence2 = original_sentence[:span_mask_start] \
                                       + sentence1['mask_token'] \
                                       + original_sentence[
                                         span_mask_end:entity_mask_start] \
                                       + replace_token \
                                       + original_sentence[entity_mask_end:]
                else:
                    masked_sentence2 = original_sentence[:entity_mask_start] \
                                       + replace_token \
                                       + original_sentence[
                                         entity_mask_end:span_mask_start] \
                                       + sentence1['mask_token'] \
                                       + original_sentence[span_mask_end:]

                masked_sentences1.append(masked_sentence1)
                masked_sentences2.append(masked_sentence2)
                target_sequences.append(sentence1['answer_str'])

            perp_loss1 = compute_perplexity(
                masked_sentences1,
                target_sequences,
                model,
                tokenizer,
                device
            )
            perp_loss2 = compute_perplexity(
                masked_sentences2,
                target_sequences,
                model,
                tokenizer,
                device
            )

            for i in range(len(batch)):

                perplexity1, loss1 = perp_loss1[i]
                perplexity2, loss2 = perp_loss2[i]

                sentence1 = batch[i]

                masked_sentence1 = masked_sentences1[i]
                masked_sentence2 = masked_sentences2[i]
                if srl_filter_results:
                    srl_filter_pass = srl_filter_results[i]
                else:
                    srl_filter_pass = 'n/a'

                if sentence1['entity_span_loc'][1] < sentence1['mask_loc'][0]:
                    distance = sentence1['mask_loc'][0] - sentence1[
                        'entity_span_loc'][1]
                else:
                    distance = sentence1['mask_loc'][1] - sentence1[
                        'entity_span_loc'][0]

                if min_sentence_len < len(sentence1['masked_sentence'].split()):
                    ex = dict()
                    ex['ex_id'] = sentence1['ex_id']
                    ex['masked_sentence_with_ent'] = masked_sentence1
                    ex['masked_sentence_without_ent'] = masked_sentence2
                    ex['answer_str'] = sentence1['answer_str']
                    ex['answer_d'] = sentence1['answer_d']
                    ex['perplexity_with_ent'] = perplexity1
                    ex['perplexity_without_ent'] = perplexity2
                    ex['nll_with_ent'] = loss1
                    ex['nll_without_ent'] = loss2
                    ex['entity_span_loc'] = sentence1['entity_span_loc']
                    ex['mask_loc'] = sentence1['mask_loc']
                    ex['srl_filter_pass'] = srl_filter_pass
                    ex['distance'] = distance
                    json.dump(ex, f)
                    f.write('\n')
                    good += 1
                    if good == max_examples:
                        break


def compute_perplexity_and_recall(examples: List[Dict],
                                  save_to: str,
                                  model: Module,
                                  tokenizer: PreTrainedTokenizerFast,
                                  device: device,
                                  batch_size: int = 1,
                                  k: int = 1,
                                  max_sentence_len: int = 1,
                                  beam_size: int = 1,
                                  mode: str = 'perplexity',
                                  n_definition: int = 0,
                                  use_random_definition: bool = False,
                                  replace_token: str = None,
                                  rng: Any = None):

    # Get all definitions
    definitions = {}
    for ex in examples:
        _pageid = ex['ex_id'].split('_')[-3]
        if _pageid not in definitions:
            definitions[_pageid] = ' '.join(ex['definition'][:n_definition]).replace('[ENT_START]', '').replace('[ENT_END]', '')

    # Create batches
    n_examples = len(examples)
    n_batches = n_examples // batch_size
    if n_examples % batch_size != 0:
        n_batches += 1
    batches = [examples[i * batch_size:(i + 1) * batch_size]
                 for i in range(n_batches)]

    bad_count = 0

    outputs = []
    for batch in tqdm(batches):

        if 'gpt' in model.name_or_path:
            masked_sentences = [ex['answer_str'] for ex in batch]
            target_sequences = [ex['answer_str'] for ex in batch]
            left_context = [ex['left_context'] for ex in batch]
            right_context = [ex['right_context'] for ex in batch]
        elif 'bart' in model.name_or_path:
            masked_sentences = [ex['masked_sentence'] for ex in batch]
            target_sequences = [ex['answer_str'] for ex in batch]
            left_context = [ex['left_context'] for ex in batch]
            right_context = [ex['right_context'] for ex in batch]
        else:  # 't5'
            masked_sentences = [ex['masked_sentence'] for ex in batch]
            target_sequences = [ex['answer_str'] for ex in batch]
            left_context = None
            right_context = None

        gold_spans = [ex['answer_d'] for ex in batch]

        if replace_token is not None:
            for sid, ms in enumerate(masked_sentences):
                ex = batch[sid]
                ent_mention = ex['entity_span_str']
                split_sentence = ms.split(ent_mention)
                # print('pre')
                # print(masked_sentences[sid])
                # print(target_sequences[sid])
                # print(left_context[sid])
                # print(right_context[sid])
                if len(split_sentence) != 2:
                    print('WARNING: bad sentence={}'.format(ex['ex_id']))
                    masked_sentences[sid] = ms.replace(
                        ent_mention, replace_token, 1)
                    bad_count += 1
                else:
                    left, right = split_sentence
                    if ex['entity_span_loc'][0] == 0:
                        masked_sentences[sid] = left + replace_token.capitalize() + right
                    else:
                        masked_sentences[sid] = left + replace_token + right

                if 'bart' in model.name_or_path:
                    tgt = target_sequences[sid]
                    split_tgt_sentence = tgt.split(ent_mention)
                    if len(split_tgt_sentence) != 2:
                        target_sequences[sid] = tgt.replace(
                            ent_mention, replace_token, 1)
                        continue
                    tgt_left, tgt_right = split_tgt_sentence
                    if ex['entity_span_loc'][0] == 0:
                        target_sequences[sid] = tgt_left + \
                                                replace_token.capitalize() + \
                                                tgt_right
                    else:
                        target_sequences[sid] = tgt_left + replace_token + \
                                                tgt_right

                    # print('post')
                    # print(masked_sentences[sid])
                    # print(target_sequences[sid])
                    # print()

                if 'gpt' in model.name_or_path:
                    target_sequences[sid] = masked_sentences[sid]
                    # Check if the entity is in the left or right context
                    if ex['entity_span_loc'][0] > len(left_context[sid].split()):
                        right_ctx = right_context[sid]
                        split_right_context = right_ctx.split(ent_mention)
                        if len(split_right_context) != 2:
                            right_context[sid] = right_ctx.replace(
                                ent_mention, replace_token, 1)
                        else:
                            ctx_left, ctx_right = split_right_context
                            if ex['entity_span_loc'][0] == 0:
                                right_context[sid] = ctx_left + \
                                                        replace_token.capitalize() + \
                                                        ctx_right
                            else:
                                right_context[sid] = ctx_left + replace_token + \
                                                        ctx_right
                    else:
                        left_ctx = left_context[sid]
                        split_left_context = left_ctx.split(ent_mention)
                        if len(split_left_context) != 2:
                            left_context[sid] = left_ctx.replace(
                                ent_mention, replace_token, 1)
                        else:
                            ctx_left, ctx_right = split_left_context
                            if ex['entity_span_loc'][0] == 0:
                                left_context[sid] = ctx_left + \
                                                        replace_token.capitalize() + \
                                                        ctx_right
                            else:
                                left_context[sid] = ctx_left + replace_token + \
                                                        ctx_right

                    # print('post')
                    # print(masked_sentences[sid])
                    # print(target_sequences[sid])
                    # print(left_context[sid])
                    # print(right_context[sid])
                    # print()

        if n_definition > 0:
            if use_random_definition is True:
                for sid, ms in enumerate(masked_sentences):
                    ex = batch[sid]
                    _pageid = ex['ex_id'].split('_')[-3]
                    candidate_definitions = [(k, v) for k,
                                                      v in definitions.items()
                                              if k != _pageid]
                    random_pageid, random_definition = rng.choice(
                        candidate_definitions)
                    masked_sentences[sid] = random_definition + ' ' + ms
                    ex['random_pageid'] = random_pageid
                    if 'bart' in model.name_or_path:
                        target_sequences[sid] = random_definition + ' ' + target_sequences[sid]
                    elif 'gpt' in model.name_or_path:
                        left_context[sid] = random_definition + ' ' + left_context[sid]

            else:
                for sid, ms in enumerate(masked_sentences):
                    ex = batch[sid]
                    gold_definition = ' '.join(ex['definition'][
                                                 :n_definition]).replace('[ENT_START]', '').replace('[ENT_END]', '')
                    masked_sentences[sid] = gold_definition + ' ' + ms
                    ex['random_pageid'] = 'n/a'
                    if 'bart' in model.name_or_path:
                        target_sequences[sid] = gold_definition + ' ' + target_sequences[sid]
                    elif 'gpt' in model.name_or_path:
                        left_context[sid] = gold_definition + ' ' + left_context[sid]
        else:
            for sid, ms in enumerate(masked_sentences):
                ex = batch[sid]
                ex['random_pageid'] = 'n/a'


        if mode == 'perplexity':
            # Get perplexity
            perp_loss = compute_perplexity(
                masked_sentences,
                target_sequences,
                model,
                tokenizer,
                device,
                left_context=left_context,
                right_context=right_context
            )
            assert len(batch) == len(perp_loss)
            for i, ex in enumerate(batch):
                output_dict = {
                    'ex_id': ex['ex_id'],
                    'masked_sentence': masked_sentences[i],
                    'answer_str': target_sequences[i],
                    'perplexity': perp_loss[i][0],
                    'loss_per_token': perp_loss[i][1],
                    'random_pageid': ex['random_pageid'],
                    'span_type': ex['span_type']
                }

                outputs.append(output_dict)
        elif mode == 'recall':
            # Get top k predictions
            pred_spans, _, _ = predict_topk(
                masked_sentences,
                target_sequences,
                model,
                tokenizer,
                device,
                k=k,
                max_sentence_len=max_sentence_len,
                beam_size=beam_size,
                left_context=left_context,
                right_context=right_context
            )
            assert len(batch) == len(pred_spans)
            for i, ex in enumerate(batch):
                em = 0
                for span_id, gold_str in gold_spans[i].items():
                    pred_strs = [_p[span_id] for _p in pred_spans[i][:k]]
                    if gold_str in pred_strs:
                        em = 1
                output_dict = {
                    'ex_id': ex['ex_id'],
                    'masked_sentence': masked_sentences[i],
                    'gold_span': gold_spans[i],
                    'pred_spans': pred_spans[i],
                    'em': em,
                    'random_pageid': ex['random_pageid'],
                    'span_type': ex['span_type']
                }
                outputs.append(output_dict)
        else:
            print(mode)
            raise NotImplementedError

    assert len(outputs) == sum([len(batch) for batch in batches])
    save_json(save_to, outputs)

    # Aggregate the results
    n_examples = len(outputs)
    n_em = 0
    nll_loss = []
    less_than_k_pred = 0
    for output in outputs:
        if mode == 'perplexity':
            nll_loss.append(np.mean([l[1] for l in output['loss_per_token']]))
        elif mode == 'recall':
            n_em += output['em']
            if len(output['pred_spans']) < k:
                less_than_k_pred += 1
        else:
            raise NotImplementedError
    print('Total {} examples.'.format(n_examples))
    if mode == 'perplexity':
        perplexity = np.exp(np.mean(nll_loss))
        print('Perplexity: {:.4f}'.format(perplexity))
    elif mode == 'recall':
        recall_k = n_em / float(n_examples)
        print('Recall@{}: {:.4f}'.format(k, recall_k))
        print('less_than_k_pred: {}'.format(less_than_k_pred))
    else:
        raise NotImplementedError

    print(f'bad_count: {bad_count} / {n_examples}')


