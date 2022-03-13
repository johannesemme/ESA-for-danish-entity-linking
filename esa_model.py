import logging
from collections import Counter
import numpy as np
import gzip
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import DanishStemmer
stemmer = DanishStemmer()
import click
import os
import re
import requests
from wikipedia2vec.dump_db import DumpDB
import pandas as pd
#from termcolor import colored

url = "http://snowball.tartarus.org/algorithms/danish/stop.txt"

logger = logging.getLogger(__name__)

WORDONWORD_THRESH = 0.4
SPANSIZE = 5

@click.command()
@click.argument("model_name", type=str)
@click.option("--stemming", default=False)
@click.option("--max_term_docfrequency", default=0.5) # ignore words/termszzzz occuring in more than half of the documents
def setup_esa_model(model_name, stemming, max_term_docfrequency, **kwargs):
    if not os.path.exists("ESA models"):
        os.makedirs("ESA models")

    dump_db = DumpDB("dump_db/dawikidump")
    ESA_model = ESA(dump_db)
    ESA_model.setup(model_name, stemming, max_term_docfrequency)

@click.command()
@click.argument("model_name", type=str)
@click.option("--num_matches", default=5)
@click.option("--extra_cand_entities", default=5)
def run_model(model_name, num_matches, extra_cand_entities):
    dump_db = DumpDB("dump_db/dawikidump")
    ESA_model = ESA(dump_db)
    ESA_model.load_pkl(model_name)

    while True:
        input_text = input("Enter query (type 'q' for exit): ")
        if input_text == "q":
            break
        print()
        ESA_model.query(query_text = input_text, n = num_matches, num_extra_ents = extra_cand_entities)
        print("------" * 20 )

class ESA(object):
    """
    Computing semantic relatedness using Wikipedia-based explicit semantic analysis.
    Explicit semantic analysis proposed by Evgeniy Gabrilovich and Shaul Markovitch, 2007.
    """
    def __init__(self, dump_db: DumpDB):
        self.dump_db = dump_db
        self.ignored_thumbs = re.compile(
                    r"""
                    (?:(?:thumb|thumbnail|left|right|\d+px|upright(?:=[0-9\.]+)?)\|)""",
                    flags=re.DOTALL | re.UNICODE | re.VERBOSE | re.MULTILINE)
        self.ignored_chars = re.compile(r"""(\"|\.\s|\:\s |\?\s |\–|\- |[()] |\,|\” |\’ |\‘ |\' )""", re.VERBOSE)
        self.snowball_stopwords = re.findall('^(\w+)', requests.get(url).text, flags=re.MULTILINE | re.UNICODE)
        self.file_path = "ESA models/"

    def danish_stemming(self, input):
        input = input.split(' ')
        return  [stemmer.stem(x) for x in input]

    def process_text(self,text):  
        text = text.lower()
        text = self.ignored_thumbs.sub('', text)
        text = self.ignored_chars.sub(' ', text)
        return text

    def iter_texts(self, tex_processing=False):
        for title in self.dump_db.titles():
            text = ""
            for par in self.dump_db.get_paragraphs(title):
                if "kategori:" not in par.text.lower():
                    if tex_processing:
                        text += self.process_text(par.text) + " "
                    else:
                        text += par.text + " "
            yield text
    

    def setup(self, model_name: str,  stemming , max_term_docfrequency):    
        
        logger.info('Collecting titles and texts')
        self._titles = list(self.dump_db.titles())
        texts = list(self.iter_texts())

        logger.info('Calculating TFIDF table')
        if stemming:
            logger.info('Stemming is performed. This might take a while (approx ~ 15 min)')
            self.model_name = model_name + "_stemmed"
            self._transformer = TfidfVectorizer(stop_words=self.snowball_stopwords, max_df=max_term_docfrequency, tokenizer=self.danish_stemming)
            self._Y = self._transformer.fit_transform(texts)    
        else:
            self._transformer = TfidfVectorizer(stop_words=self.snowball_stopwords, max_df=max_term_docfrequency)
            self._Y = self._transformer.fit_transform(texts) 

        logger.info('Saving model to pickle...')
        self.save_pkl(model_name)

    def save_pkl(self, model_name):
        try:
            os.makedirs(self.file_path)
        except FileExistsError:
            pass 
        
        items = [
            ('_titles', f'esa-{model_name}-titles.pkl.gz'),
            ('_Y', f'esa-{model_name}-y.pkl.gz'),
            ('_transformer', f'esa-{model_name}-transformer.pkl.gz')
        ]
        for attr, filename in items:
            full_filename = self.file_path + filename
            logger.info('Writing parameters to pickle file {}'.format(full_filename))
            with gzip.open(full_filename, 'w') as f:
                pickle.dump(getattr(self, attr), f, -1)

    def load_pkl(self, model_name):
        self.model_name = model_name
        logger.info('Loading model...')
        items = [
            ('_titles', f'esa-{model_name}-titles.pkl.gz'),
            ('_Y', f'esa-{model_name}-y.pkl.gz'),
            ('_transformer', f'esa-{model_name}-transformer.pkl.gz')
        ]
        for attr, filename in items:
            full_filename = self.file_path + filename
            logger.info('Reading parameters from pickle file {}'.format(full_filename))
            with gzip.open(full_filename) as f:
                setattr(self, attr, pickle.load(f))

    def relatedness(self, phrases):
        Y = self._transformer.transform(phrases)
        D = np.asarray((self._Y * Y.T).todense())
        D = np.einsum('ij,j->ij', D,
                      1 / np.sqrt(np.multiply(D, D).sum(axis=0)))
        return D.T.dot(D)

    @staticmethod
    def get_entitiyspans(text, span_size):
        word_start_positions = [0]
        word_end_positions = []
        for m in re.finditer(" \w", text):
            word_start_positions.append(m.end(0)-1)
            word_end_positions.append(m.start(0))
        word_end_positions = word_end_positions + [len(text)]

        entity_spans = []
        for i, start_pos in enumerate(word_start_positions):
            for j, end_pos in enumerate(word_end_positions[i:]):
                if j == span_size:
                    break
                entity_spans.append((start_pos, end_pos))
        
        return entity_spans

    @staticmethod
    def jaccard_similarity(seq1, seq2):
        list1 = seq1.lower().split()
        list2 = seq2.lower().split()
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    @staticmethod
    def wordOnword(men,cand):
        men_list,cand_list = men.lower().split(),cand.lower().split()
        word_num = min(len(men_list),len(cand_list))
        cnt = 0
        for word in range(word_num):
            men_word = men_list[word]
            cand_word = cand_list[word]
            min_char_len = min(len(men_word), len(cand_word))
            for char in range(min_char_len):
                if men_word[char] == cand_word[char]:
                    cnt += 1          
        norm = max(len(men.replace(" ", "")),len(cand.replace(" ", "")))
        return cnt / norm


    @staticmethod
    def overlap(tuples, search):
        res = []
        for t in tuples:
            if(t[1]>search[0] and t[0]<search[1]):
                res.append(t)
        return res



    def query(self, query_text, n, num_extra_ents):
        
        # 1) Process the query
        processed_query_text = self.ignored_chars.sub(' ', query_text)
        if "stemmed" in self.model_name:
            processed_query_text = " ".join([w for w in self.danish_stemming(processed_query_text)])

        # 2) Get at candidate pool dictionary with top ESA candidates sorte by score
        y = self._transformer.transform([processed_query_text]) # transform query text with TFIDF transformer
        D = np.array((self._Y * y.T).todense()) # calculate score using dot product
        indices = np.argsort(-D, axis=0)
        cand_entities = [self._titles[index] for index in indices[:n, 0]]
        scores = [D[index][0] for index in indices[:n, 0]]
        candidate_pool = dict(zip(cand_entities, scores))

        print(candidate_pool)
