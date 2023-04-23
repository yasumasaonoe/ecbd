import argparse
import json
import os
import wptools
import html
import spacy
import re
import requests
import dateutil
import multiprocessing
import random
import inflect

from tqdm import tqdm
from typing import Any, List, Tuple
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--enwiki_dir",
                    help="Root dir for parsed EN Wiki pages.",
                    default="")
parser.add_argument("--target_pageids",
                    help="List of the target pageids.  This must be in the "
                         "txt format.",
                    default="")
parser.add_argument("--cached_wiki_pages",
                    help="Cached EN Wikipedia pages.",
                    default="")
parser.add_argument("--save_to",
                    help="A JSON file that sentences will be saved.",
                    default="")
parser.add_argument("--top_paragraphs",
                    help="Number of paragraphs to keep.",
                    default=None,
                    type=int)
parser.add_argument("--lowercase",
                    help="Do lowercase.",
                    default=False,
                    type=bool)
parser.add_argument("--add_aliases",
                    help="Add aliases.",
                    default=True,
                    type=bool)
parser.add_argument("--plural",
                    help="Include plural of entities.",
                    default=False,
                    type=bool)
parser.add_argument("--split_name",
                    help="Split entity name.",
                    default=False,
                    type=bool)
parser.add_argument("--min_sentence_length",
                    help="Minimum sentence length.",
                    default=5,
                    type=int)
parser.add_argument("--use_wptools",
                    help="Use Wptools.",
                    default=False,
                    type=bool)

NLP = spacy.load("en_core_web_sm")
NLP.add_pipe('sentencizer')
PLURAL = inflect.engine()


def get_ids_and_text(
        dir_path: str,
        entities: List[str] = None
) -> List[Tuple[int, str, str]]:
    all_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        all_filenames += [os.path.join(dirpath, file) for file in filenames]
    all_filenames = sorted(all_filenames)
    pageinfo = []
    entities = set(entities)
    for filename in tqdm(all_filenames):
        with open(filename) as f:
            for line in f:
                page = json.loads(line.strip())
                if entities is not None:
                    if page['id'] in entities:
                        pageinfo.append((page['id'], page['title'], page['text']))
                else:
                    pageinfo.append((page['id'], page['title'], page['text']))
    return pageinfo


def preprocess_text(text: str):
    s = html.unescape(text)
    s = s.replace('<bold>', '[ENT_START]').replace('</bold>', '[ENT_END]')
    s = s.replace('<boldi>', '[ENT_START]').replace('</boldi>', '[ENT_END]')
    s = re.sub('<[^>]*>', '', s)
    # s = re.sub(r"\n+", "\n", s)
    # s = s.replace("\n", " ").strip()
    s = s.split('\n')
    return [p for p in s if len(p.split()) > 5]


class WikiPage:

    def __init__(self,
                 pageid: str,
                 title: str,
                 text: str,
                 wiki_metadata: dict = None,
                 use_wptools: bool = True):
        self.title = title
        self.pageid = pageid
        self.page = None
        self.aliases = [title]
        if pageid in wiki_metadata:
            if 'aliases' in wiki_metadata[pageid]['metadata']:
                self.page = wiki_metadata[pageid]['metadata']
                self.aliases = self.page['aliases']
        elif use_wptools:
            try:
                page = wptools.page(pageid=pageid, silent=True)
                page.get_query()
                self.page = page.data
                self.aliases = page.data['aliases']
            except:
                print(f'LookupError: {pageid}')
                pass
        self.raw_text = text
        self.paragraphs = preprocess_text(text)
        self.docs = [NLP(p) for p in self.paragraphs]
        self.definition = [s.text for s in self.docs[0].sents] \
            if self.docs else []

    def __repr__(self):
        return self.title

    def get_sentences(
            self,
            top_paragraphs: int = None,
            lowercase: bool = False,
            add_aliases: bool = False,
            plural: bool = False,
            split_name: bool = False,
            mask_definition :bool = False
    ) -> List[Tuple[Tuple[int, int], str, Any]]:
        if top_paragraphs is None:
            top_paragraphs = len(self.docs)
        # search title -> singular -> plural
        aliases = [self.title, self.title.lower()] \
            if lowercase else [self.title]
        if split_name:
            aliases = [s for a in aliases if len(a.split()) == 2 and \
                       all(not w.islower() for w in a.split())
                       for s in a.split()]
        if add_aliases:
            aliases += self.aliases + [a.lower() for a in self.aliases] if \
                lowercase else self.aliases
        if plural:
            aliases += [PLURAL.plural(a) for a in aliases]
        max_spans = 1  # limit num of spans up to 1
        sents = []
        paraid = 0
        for doc in self.docs[:top_paragraphs]:
            sentid = 0
            if paraid > 0:
                break
            for sent in doc.sents:
                if paraid == 0:
                    for ent in aliases:
                        try:
                            found = list(
                                re.finditer('\\b' + ent + '\\b', sent.text))
                        except:
                            print('regex error:', ent, ': ', sent.text)
                            continue
                        if 0 < len(found) <= max_spans:
                            # double check
                            if ent.islower():
                                if sent.text.lower().count(ent) <= max_spans:
                                    for sid in range(len(found)):
                                        sents.append(
                                            (
                                                found[sid].span(0),
                                                found[sid].group(0),
                                                sent
                                            )
                                        )
                                    break
                            else:
                                if sent.text.count(ent) <= max_spans:
                                    for sid in range(len(found)):
                                        sents.append(
                                            (
                                                found[sid].span(0),
                                                found[sid].group(0),
                                                sent
                                            )
                                        )
                                    break
                        else:
                            sents.append(
                                (
                                    [-1, -1],
                                    'ENTITY_NOT_FOUND',
                                    sent
                                )
                            )
                            break

                sentid += 1
            paraid += 1
        return sents


def main():
    args = parser.parse_args()

    # Get target pageids.
    if args.target_pageids:
        with open(args.target_pageids) as f:
            target_pageids = [l.strip() for l in f]
    else:
        target_pageids = None

    # Get parsed text for the target entities.
    enwiki_ids_text = get_ids_and_text(args.enwiki_dir,
                                       entities=target_pageids)

    # Use cached wiki page meta data (this is faster).
    if args.cached_wiki_pages:
        with open(args.cached_wiki_pages) as f:
            cached_wiki_pages = [json.loads(l.strip()) for l in f]
            cached_wiki_pages = {cw['pageid']: cw for cw in
                                 cached_wiki_pages}
    else:
        cached_wiki_pages = None

    n_entities = 0
    n_bad_entities = 0
    with open(args.save_to, 'w') as f:
        for cp in tqdm(enwiki_ids_text):
            wp = WikiPage(*cp,
                          wiki_metadata=cached_wiki_pages,
                          use_wptools=args.use_wptools)
            n_entities += 1
            if wp.aliases and wp.definition:
                sents = wp.get_sentences(
                    top_paragraphs=args.top_paragraphs,
                    lowercase=args.lowercase,
                    add_aliases=args.add_aliases,
                    plural=args.plural,
                    split_name=args.split_name
                )
                assert wp.title == cp[1], (wp.title, cp[1])
                sentences = []
                for sent in sents:
                    if len(sent[2].text.split()) > args.min_sentence_length:
                        sentences.append(
                            {
                                'sentence': sent[2].text,
                                'ent_loc': sent[0],
                                'ent_str': sent[1]
                            }
                        )
                d = {
                    'pageid': wp.pageid,
                    'title': wp.title,
                    'sentences': sentences,
                    'definition': wp.definition
                }
                json.dump(d, f)
                f.write('\n')
            else:
                n_bad_entities += 1
    print(f'#Entities   : {n_entities}')
    print(f'Lookup Error: {n_bad_entities}')


if __name__ == '__main__':
    main()
