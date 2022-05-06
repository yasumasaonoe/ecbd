"""
Utils for data preprocessing

"""


import json
import re
import spacy

from collections import Counter, defaultdict
from spacy.tokens import Doc, Span
from tqdm import tqdm
from typing import Any, Dict, List, TextIO, Tuple


NLP = spacy.load("en_core_web_sm")
PRINTABLE_ASCII_SYMBOLS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'

NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS']


def load_jason(file_path):
    with open(file_path) as f:
        return [json.loads(l.strip()) for l in f]


def save_json(file_path, data):
    with open(file_path, 'w') as f:
        for ex in data:
            json.dump(ex, f)
            f.write('\n')


def merge_phrases_in_doc(doc: Doc) -> Doc:
    with doc.retokenize() as retokenizer:
        for np in list(doc.noun_chunks):
            attrs = {
                "tag": np.root.tag_,
                "lemma": np.root.lemma_,
                "ent_type": np.root.ent_type_,
            }
            retokenizer.merge(np, attrs=attrs)
    return doc


def get_entity_span(doc: Doc, entity_span_str: str) -> List[Span]:
    candidates = []
    for nc in doc.noun_chunks:
        if nc.text == entity_span_str or entity_span_str in nc.text:
            candidates.append(nc)
    return candidates


def get_noun_phrases(
        doc: Doc,
        entity_span_obj: Span,
        exceptions: List[str] = None,
        include_named_entities: bool = False,
) -> List[Tuple[Span, str]]:
    if entity_span_obj is None:
        return []
    selected_spans = []
    seen_spans = set()
    if include_named_entities:
        # Remove the key entity
        named_entities = [span for span in doc.ents
                          if not is_overlapping(span, entity_span_obj)]
        selected_spans.extend([(span, 'PROPER') for span in named_entities])
        seen_spans = set([(span.text, span.start, span.end) for span in
                          named_entities])
    for nc in doc.noun_chunks:
        if not is_overlapping(nc, entity_span_obj):
            for np in [nc, doc[nc.root.left_edge.i:nc.root.right_edge.i + 1]]:
                span_info = (np.text, np.start, np.end)
                if span_info not in seen_spans:
                    if is_acceptable_noun_span(np, exceptions=exceptions):
                        phrase_type = 'PROPER' if is_proper_noun_phrase(np) \
                            else 'COMMON'
                        selected_spans.append((np, phrase_type))
                        seen_spans.add(span_info)
    return remove_overlapping_spans(selected_spans)


def is_acceptable_noun_span(span: Span, exceptions: List[str] = None) -> bool:
    # Incomplete parentheses, punctuations, quotes
    if (span[0].pos_ == 'PUNCT' and span[-1].pos_ != 'PUNCT' and [
        s.pos_ for s in span].count('PUNCT') == 1) or \
       (span[0].pos_ != 'PUNCT' and span[-1].pos_ == 'PUNCT' and [
        s.pos_ for s in span].count('PUNCT') == 1):
        return False
    # Max span length
    if len(span) > 5:
        return False
    # Symbols
    if all(s.text in PRINTABLE_ASCII_SYMBOLS for s in span):
        return False
    # Pronouns
    if all(s.pos_ == 'PRON' for s in span):
        return False
    # Exception words
    if exceptions is not None:
        for ex_word in exceptions:
            if span.text == ex_word or ex_word in span.text:
                return False
    return True


def is_proper_noun_phrase(span: Span) -> bool:
    # TODO: a bit conservative now.
    # All words are PROPN
    if all(w.pos_ == 'PROPN' for w in span):
        return True
    # The last word is PROPN
    if span[-1].pos_ == 'PROPN':
        return True
    # DET + PROPN + other
    if len(span) > 1 and span[0].pos_ == 'DET' and span[1].pos_ == 'PROPN':
        return True
    # False otherwise...
    return False


def remove_overlapping_spans(
        spans: List[Tuple[Span, str]]
) -> List[Tuple[Span, str]]:
    to_remove_ids = set()
    n_spans = len(spans)
    # Take a span with a determiner
    for i in range(n_spans):
        for j in range(n_spans):
            if i != j:
                span1 = spans[i][0]
                span2 = spans[j][0]
                if span1.start < span2.start and span2.end == span1.end:
                    if span1[0].pos_ == 'DET' and span1[1:].text == span2.text:
                        to_remove_ids.add(j)
    return [span for k, span in enumerate(spans) if k not in to_remove_ids]


def is_overlapping(span1: Span, span2: Span) -> bool:
    return (span1.start <= span2.start <= span1.end) \
           or (span2.start <= span1.start <= span2.end)


def clean_span(span: str):
    cleaned_span = span
    return cleaned_span


def clean_sentence(sentence: str):
    # Drop entity tags
    cleaned_sentence = sentence.strip()
    cleaned_sentence = cleaned_sentence.replace(
        '[ENT_START]', '').replace('[ENT_END]', '')
    cleaned_sentence = cleaned_sentence.replace("(')", '')
    cleaned_sentence = re.sub('\s+', ' ', cleaned_sentence)
    return cleaned_sentence


def get_random_spans(
        doc: Doc,
        entity_span_obj: Span,
        rng: Any,
        n_spans: int = 3,
        distance: int = 10,
        max_span_length: int = 5,
) -> List[Tuple[Span, str]]:
    if entity_span_obj is None:
        return []
    spans = set()
    sentence_length = len(doc)
    ent_start = entity_span_obj.start
    ent_end = entity_span_obj.end - 1
    ent_span = list(range(ent_start, ent_end + 1))
    min_loc = max(0, ent_start - distance)
    max_loc = min(len(doc), ent_end + distance)
    start_positions = [i for i in range(min_loc, max_loc) if i not in ent_span]
    lengths = list(range(1, max_span_length + 1))
    n_trials = 0
    while len(spans) < n_spans:
        if n_trials == 5000 and len(spans) < n_spans:
            break
        n_trials += 1
        length = rng.choice(lengths)
        span_start = rng.choice(start_positions)
        span_end = span_start + length
        if span_end > sentence_length - 1:
            continue
        if set(ent_span).intersection(set(list(range(span_start, span_end + 1)))):
            continue
        else:
            spans.add((doc[span_start: span_end + 1], 'RANDOM'))
    return list(spans)

def get_span_location(
        sentence: str,
        span_str: str
) -> Tuple[int, int]:
    assert span_str in sentence
    split_sentence = [chunk.strip() for chunk in sentence.split(span_str)]
    assert len(split_sentence) == 2, (span_str, sentence)
    entity_start = len(split_sentence[0].split())
    entity_end = entity_start + len(span_str.split())
    return entity_start, entity_end


def is_in_quotes(doc: Doc, entity_span_obj: Span) -> bool:
    pattern = re.compile(r'"([^"]*)"')
    entity_start = entity_span_obj.start_char
    entity_end = entity_span_obj.end_char - 1
    for matched in pattern.finditer(doc.text):
        span_start = matched.start()
        span_end = matched.end() - 1
        if (span_start <= entity_start <= span_end) \
                or (entity_start <= span_start <=  entity_end):
            return True
    return False


class Sentence:

    def __init__(self,
                 sentence: str,
                 entity_id: str,
                 entity_name: str,
                 entity_span_str: str,
                 entity_span_loc: List[int],  # char loc
                 definition: List[str] = None,
                 mask_token: str = '<extra_id_{}>',
                 mask_type: str = 'noun_phrases',
                 merge_phrases: bool = False,
                 include_named_entities: bool = False,
                 rng: Any = None):
        self.original_sentence = sentence
        self.doc = NLP(sentence)
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.entity_span_str = entity_span_str
        self.entity_span_loc_char = entity_span_loc
        self.entity_span_obj = self.doc.char_span(*entity_span_loc)
        self.entity_span_loc_word = (
            self.entity_span_obj.start, self.entity_span_obj.end) if \
            self.entity_span_obj is not None else None
        self.definition = definition
        self.mask_token = mask_token
        if merge_phrases:
            self.doc = merge_phrases_in_doc(self.doc)
        if mask_type == 'noun_phrases':
            self.mask_spans = get_noun_phrases(
                self.doc,
                self.entity_span_obj,
                exceptions=None,
                include_named_entities=include_named_entities
            )
        elif mask_type == 'random':
            if rng is None:
                raise ValueError
            # TODO: remove hardcoded args
            self.mask_spans = get_random_spans(
                self.doc,
                self.entity_span_obj,
                rng,
                n_spans=5,
                distance=10,
                max_span_length=5
            )
        else:
            raise NotImplementedError
        self.sentence = self.doc.text

    def __repr__(self):
        return self.doc.text

    def get_all_masked_sentences(self,
                                 prepend_str: str = None,
                                 fake_entity: str = None):

        all_masked_sentences = []

        for mask_span, span_type in self.mask_spans:

            assert mask_span.text in self.sentence and \
                   self.entity_span_str in self.sentence, (
                self.sentence, mask_span.text, self.entity_span_str)

            start_loc_char = mask_span.start_char
            end_loc_char = mask_span.end_char

            masked_sentence = self.sentence[:start_loc_char] \
                              + self.mask_token.format(0) \
                              + self.sentence[end_loc_char:]

            if fake_entity is not None:
                masked_sentence = masked_sentence.replace(self.entity_span_str,
                                                          fake_entity)

            answer_str = '{} {} {}'.format(self.mask_token.format(0),
                                           mask_span.text,
                                           self.mask_token.format(1))
            answer_d = {self.mask_token.format(0): mask_span.text}

            # Count the mask token (it should be 1).
            n_masks = masked_sentence.count(self.mask_token.format(0))
            if n_masks != 1:
                # Got multiple masks...
                print('WARNING:', n_masks, mask_span.text, masked_sentence)
                continue

            if prepend_str is not None:
                masked_sentence = ' '.join([prepend_str, masked_sentence])

            masked_sentence = clean_sentence(masked_sentence)

            mask_loc = get_span_location(
                masked_sentence, self.mask_token.format(0))

            all_masked_sentences.append(
                (
                    self.sentence,
                    masked_sentence,
                    answer_str, answer_d,
                    self.entity_span_loc_word,
                    mask_loc,
                    (start_loc_char, end_loc_char),
                    self.entity_span_str,
                    mask_span.text,
                    self.mask_token.format(0),
                    span_type
                )
            )

        return all_masked_sentences

    def mask_entity(self):

        noun_phrase_including_entity = [nc for nc in self.doc.noun_chunks if
                                        nc.start <= self.entity_span_obj.start
                                        and self.entity_span_obj.end <= nc.end]
        if len(noun_phrase_including_entity) <= 1:
            if len(noun_phrase_including_entity) == 1:
                start_loc_char = noun_phrase_including_entity[0].start_char
                end_loc_char = noun_phrase_including_entity[0].end_char
            else:  # length = 0
                start_loc_char = self.entity_span_obj.start_char
                end_loc_char = self.entity_span_obj.end_char
            masked_sentence = self.sentence[:start_loc_char] \
                              + self.mask_token.format(0) \
                              + self.sentence[end_loc_char:]
        else:
            # This shouldn't happen
            print(self.sentence)
            print('ERROR',
                  noun_phrase_including_entity,
                  list(self.doc.noun_chunks),
                  self.entity_span_obj)
            raise
        # Count the mask token (it should be 1).
        n_masks = masked_sentence.count(self.mask_token.format(0))
        if n_masks != 1:
            # Got multiple masks...
            print('WARNING:', n_masks, self.entity_span_str, masked_sentence)
        masked_sentence = clean_sentence(masked_sentence)
        answer_str = '{} {} {}'.format(self.mask_token.format(0),
                                       self.entity_span_str,
                                       self.mask_token.format(1))
        answer_d = {self.mask_token.format(0): self.entity_span_str}
        mask_loc = get_span_location(
            masked_sentence, self.mask_token.format(0))
        return self.sentence, masked_sentence, answer_str, answer_d, \
               self.entity_span_loc_word, mask_loc, \
               (start_loc_char, end_loc_char), \
               self.entity_span_str, self.entity_span_str, \
               self.mask_token.format(0), 'ENTITY'


def format_input_data(file_path: str,
                      save_to: str,
                      masking_mode: str = 'span',
                      mask_type: str = 'noun_phrases',
                      rng: Any = None):

    def write_jsonline(f: TextIO,
                       ex_id: str,
                       original_sentence: str,
                       masked_sentence: str,
                       answer_str: str,
                       answer_d: Dict[str, str],
                       entity_span_loc: Tuple[int, int],
                       mask_loc: Tuple[int, int],
                       masked_span_loc_char: Tuple[int, int],
                       entity_span_str: str,
                       mask_span_str: str,
                       mask_token: str,
                       span_type: str):
        d = {
            'ex_id': ex_id,
            'original_sentence': original_sentence,
            'masked_sentence': masked_sentence,
            'answer_str': answer_str,
            'answer_d': answer_d,
            'entity_span_loc': entity_span_loc,
            'mask_loc': mask_loc,
            'masked_span_loc_char': masked_span_loc_char,
            'entity_span_str': entity_span_str,
            'mask_span_str': mask_span_str,
            'mask_token': mask_token,
            'span_type': span_type
        }
        json.dump(d, f)
        f.write('\n')

    entities = load_jason(file_path)

    entity_span_not_found = 0
    n_total_sentences = 0

    with open(save_to, 'w') as fw:
        for entity in tqdm(entities):
            pageid = entity['pageid']
            title = entity['title']
            for sent_id, sent in enumerate(entity['sentences']):
                n_total_sentences += 1
                sentence = sent['sentence']
                ent_str = sent['ent_str']
                ent_loc = sent['ent_loc']
                sentence_obj = Sentence(
                    sentence,
                    pageid,
                    title,
                    ent_str,
                    ent_loc,
                    definition=None,
                    merge_phrases=False,
                    mask_token='<extra_id_{}>',
                    mask_type=mask_type,
                    rng=rng
                )
                # Entity span can't be parsed properly
                if sentence_obj.entity_span_obj is None:
                    entity_span_not_found += 1
                    continue
                # Entity span is in quotes
                if is_in_quotes(sentence_obj.doc, sentence_obj.entity_span_obj):
                    continue
                if masking_mode.startswith('span'):
                    # Mask out spans.
                    if masking_mode == 'span':
                        masked_sentences =  \
                            sentence_obj.get_all_masked_sentences()
                    elif masking_mode == 'span_w_fake_entity':
                        fake_entity = sent['fake_entity']
                        masked_sentences =  \
                            sentence_obj.get_all_masked_sentences(
                                fake_entity=fake_entity)
                    else:
                        raise NotImplementedError
                    span_id = 0
                    for original_sentence, masked_sentence, answer_str, \
                        answer_d, entity_span_loc, mask_loc,  \
                        masked_span_loc_char, entity_span_str, mask_span_str, \
                        mask_token, span_type in masked_sentences:
                        ex_id = '{}_{}_{}_{}'.format(
                            title, pageid, sent_id, span_id)
                        write_jsonline(
                            fw,
                            ex_id,
                            original_sentence,
                            masked_sentence,
                            answer_str,
                            answer_d,
                            entity_span_loc,
                            mask_loc,
                            masked_span_loc_char,
                            entity_span_str,
                            mask_span_str,
                            mask_token,
                            span_type
                        )
                        span_id += 1
                elif masking_mode == 'entity':
                    # Mask out entities.
                    ex_id = '{}_{}_{}_0'.format(title, pageid, sent_id)
                    original_sentence, masked_sentence, answer_str, answer_d, \
                    entity_span_loc, mask_loc, masked_span_loc_char, \
                    entity_span_str, mask_span_str, mask_token, span_type = \
                        sentence_obj.mask_entity()
                    write_jsonline(
                        fw,
                        ex_id,
                        original_sentence,
                        masked_sentence,
                        answer_str,
                        answer_d,
                        entity_span_loc,
                        mask_loc,
                        masked_span_loc_char,
                        entity_span_str,
                        mask_span_str,
                        mask_token,
                        span_type
                    )
                else:
                    raise NotImplementedError

    print(f'{entity_span_not_found} / {n_total_sentences}')


def split_dataset(file_path: str,
                  save_to: str,
                  definition_file: str,
                  span_file: str,
                  entity_file_path: str = None,
                  distance: int = 10,
                  min_n_sentences: int = 1,
                  max_n_sentences: int = 100,
                  rng: Any = None):

    # Load files
    definitions = load_jason(definition_file)
    definitions = {e['pageid']: e['definition'] for e in definitions}
    span_info = load_jason(span_file)
    span_info = {s['ex_id']:s for s in span_info}
    examples = load_jason(file_path)

    # Get all entity ids
    if entity_file_path is not None:
        with open(entity_file_path) as f:
            entities = json.load(f)
        test_entities = set([ent['pageid'] for ent in entities['test']])
        dev_entities = set([ent['pageid'] for ent in entities['dev']])
    else:
        # Use all entities as a dev set
        test_entities = set()
        dev_entities = set()

    test_examples = defaultdict(list)
    dev_examples = defaultdict(list)
    for ex in tqdm(examples):
        pageid = ex['ex_id'].split('_')[-3]
        if distance >= ex['distance']:
            _span = span_info[ex['ex_id']]
            assert ex['answer_str'] == _span['answer_str'], (
                ex['ex_id'], ex['answer_str'], _span['answer_str'])
            _ex = {
                'ex_id': ex['ex_id'],
                'masked_sentence': ex['masked_sentence_with_ent'],
                'answer_str': ex['answer_str'],
                'answer_d': ex['answer_d'],
                'span_type': _span['span_type'],
                'definition': definitions[pageid]
            }
            if entity_file_path is not None:
                if pageid in test_entities:
                    test_examples[pageid].append(_ex)
                elif pageid in dev_entities:
                    dev_examples[pageid].append(_ex)
                else:
                    print(ex['ex_id'])
                    raise ValueError
            else:
                test_examples[pageid].append(_ex)

    test_final = []
    for pageid, sentences in test_examples.items():
        if len(sentences) < min_n_sentences:
            continue
        elif len(sentences) > max_n_sentences:
            sampled = rng.sample(sentences, max_n_sentences)
            test_final.extend(sampled)
        else:
            test_final.extend(sentences)

    idx = list(range(len(test_final)))
    rng.shuffle(idx)
    test_final = [test_final[i] for i in idx]

    dev_final = []
    for pageid, sentences in dev_examples.items():
        if len(sentences) < min_n_sentences:
            continue
        elif len(sentences) > max_n_sentences:

            sampled = rng.sample(sentences, max_n_sentences)
            dev_final.extend(sampled)
        else:
            dev_final.extend(sentences)

    idx = list(range(len(dev_final)))
    rng.shuffle(idx)
    dev_final = [dev_final[i] for i in idx]

    print('# Test: {}'.format(len(test_final)))
    print('# Dev  : {}'.format(len(dev_final)))
    save_json(save_to + '_test.json', test_final)
    save_json(save_to + '_dev.json', dev_final)


def count_sentences_per_ent(data):
    return sorted(
        Counter([ex['ex_id'].split('_')[-3] for ex in data]).items(),
        key=lambda x: x[1],
        reverse=True
    )


def format_example_for_bart(ex: Dict) -> Dict:
    masked_sentence = ex['masked_sentence'].split('<extra_id_0>')
    assert len(masked_sentence) == 2
    left_context, right_context = masked_sentence
    new_ex = {
        'ex_id': ex['ex_id'],
        'masked_sentence': ex['masked_sentence'].replace(
            '<extra_id_0>', '<mask>'),
        'answer_str': ex['masked_sentence'].replace(
            '<extra_id_0>', ex['answer_d']['<extra_id_0>']),
        'left_context': left_context.strip(),  # Let a model predict ' ' together
        'right_context': right_context,
        'answer_d': ex['answer_d'],
        'span_type': ex['span_type'],
        'definition': ex['definition'],
        'year': ex['year']
    }

    if 'entity_span_str' in ex:
        new_ex['entity_span_str'] = ex['entity_span_str']
    if 'entity_span_loc' in ex:
        new_ex['entity_span_loc'] = ex['entity_span_loc']

    return new_ex


def format_example_for_gpt(ex, pad_token):
    masked_sentence = ex['masked_sentence'].split('<extra_id_0>')
    assert len(masked_sentence) == 2
    left_context, right_context = masked_sentence
    new_ex = {
        'ex_id': ex['ex_id'],
        'masked_sentence': ex['masked_sentence'] + ' ' + pad_token,  # avoid an error
        'answer_str': ex['masked_sentence'].replace('<extra_id_0>', ex['answer_d']['<extra_id_0>']),
        'left_context': left_context.strip(),  # Let model predict ' ' together
        'right_context': right_context + ' ' + pad_token,  # avoid an error
        # ' ' after the span should be in the right context
        'answer_d': ex['answer_d'],
        'span_type': ex['span_type'],
        'definition': ex['definition'],
        'year': ex['year']
    }

    if 'entity_span_str' in ex:
        new_ex['entity_span_str'] = ex['entity_span_str']
    if 'entity_span_loc' in ex:
        new_ex['entity_span_loc'] = ex['entity_span_loc']

    return new_ex

if __name__ == '__main__':

    # Test
    #s = 'In 1908, Tolstoy wrote "A Letter to a Hindu" outlining his belief
    # in non-violence as a means for India to gain independence from colonial rule.'
    s = 'However, attempts to sell Gulf Oil (Great Britain) to KPC failed ' \
        'because of irrevocable GOC guarantees given earlier in regard to bonds issued to finance the construction of refinery facilities in the UK.'
    obj = Sentence(
        s,
        '123',
        'Gulf Oil',
        'Gulf Oil',
        [26, 34],
        definition=None,
        mask_token='<extra_id_{}>',
        merge_phrases=False,
        include_named_entities=True
    )
    print(obj.mask_spans)
    print(obj.entity_span_obj)
    print()
    print(obj.mask_entity())
    print()
    for ms in obj.get_all_masked_sentences():
        print(ms)
    print()
    for ms in obj.get_all_masked_sentences(fake_entity='it'):
        print(ms)



