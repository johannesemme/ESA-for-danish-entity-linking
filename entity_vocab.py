from typing import List, Dict
import json
import os
import multiprocessing
from collections import Counter, OrderedDict, defaultdict, namedtuple
from contextlib import closing
from multiprocessing.pool import Pool

import click
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB


Entity = namedtuple("Entity", "title")

_dump_db = None  # global variable used in multiprocessing workers

@click.command()
@click.option("--vocab-size", default=1000000)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
def build_entity_vocab(**kwargs):

    if not os.path.exists("ent_vocab"):
        os.makedirs("ent_vocab")

    dump_db = DumpDB("dumps/da_dump")
    EntityVocab.build(dump_db, out_file="ent_vocab/entity-vocab.jsonl", **kwargs)


class EntityVocab(object):
    def __init__(self, vocab_file: str):
        self._vocab_file = vocab_file

        self.vocab: Dict[Entity, int] = {}
        self.counter: Dict[Entity, int] = {}
        self.inv_vocab: Dict[int, List[Entity]] = defaultdict(list)

        self._parse_jsonl_vocab_file(vocab_file)

    def _parse_jsonl_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            entities_json = [json.loads(line) for line in f]

        for item in entities_json:
            title = item["title"]
            entity = Entity(title)
            self.vocab[entity] = item["id"]
            self.counter[entity] = item["count"]
            self.inv_vocab[item["id"]].append(entity)

    @property
    def size(self) -> int:
        return len(self)

    def __reduce__(self):
        return (self.__class__, (self._vocab_file,))

    def __len__(self):
        return len(self.inv_vocab)

    def __contains__(self, item: str):
        return self.contains(item)

    def __getitem__(self, key: str):
        return self.get_id(key)

    def __iter__(self):
        return iter(self.vocab)

    def contains(self, title: str):
        return Entity(title) in self.vocab

    def get_id(self, title: str, default: int = None) -> int:
        try:
            return self.vocab[Entity(title)]
        except KeyError:
            return default

    def get_title_by_id(self, id_: int) -> str:
        for entity in self.inv_vocab[id_]:
            return entity.title

    def get_count_by_title(self, title: str) -> int:
        entity = Entity(title)
        return self.counter.get(entity, 0)

    def get_total_links(self) -> int:
        cnt = 0
        for title in self.vocab:
            entity = Entity(title)
            cnt += self.counter.get(entity.title, 0)
        return cnt

    @staticmethod
    def build(
        dump_db: DumpDB,
        out_file: str,
        vocab_size: int,
        pool_size: int,
        chunk_size: int,
    ):
        counter = Counter()
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            with closing(Pool(pool_size, initializer=EntityVocab._initialize_worker, initargs=(dump_db,))) as pool:
                for ret in pool.imap_unordered(EntityVocab._count_entities, dump_db.titles(), chunksize=chunk_size):
                    counter.update(ret)
                    pbar.update()

        title_dict = OrderedDict()

        valid_titles = frozenset(dump_db.titles())
        for title, count in counter.most_common():
            if title in valid_titles and title.startswith("Kategori:"):
                title_dict[title] = count
                if len(title_dict) == vocab_size:
                    break

        ents_with_incoming_link = set(title_dict.keys())
        ents_no_incoming_link = valid_titles - ents_with_incoming_link

        for e in ents_no_incoming_link:
            title_dict[e] = 0

        with open(out_file, "w") as f:
            for ent_id, (title, count) in enumerate(title_dict.items()):
                json.dump({"id": ent_id, "title": title, "count": count}, f) 
                f.write("\n")

    @staticmethod
    def _initialize_worker(dump_db: DumpDB):
        global _dump_db
        _dump_db = dump_db

    @staticmethod
    def _count_entities(title: str) -> Dict[str, int]:
        counter = Counter()
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                title = _dump_db.resolve_redirect(wiki_link.title)
                counter[title] += 1
        
        return counter

