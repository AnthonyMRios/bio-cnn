import re
import random
import numpy as np
import gensim
import nltk
nltk.download('punkt')

def load_data_file(txt_filename, tgt_filename):
    txt = open(txt_filename, 'r')
    tgt = open(tgt_filename, 'r')
    X_txt = []
    Y = []
    for txt_line, tgt_line in zip(txt, tgt):
        X_txt.append(txt_line.strip())
        Y.append(tgt_line.strip())
    return X_txt, Y

class ProcessData(object):
    def __init__(self, pretrain_wv=None, lower=True, min_df=5):
        self.pattern = re.compile(r'(?u)\b\w\w+\b')
        self.min_df = min_df
        self.lower = lower
        if pretrain_wv is not None:
            self.wv = gensim.models.Word2Vec.load(pretrain_wv)
        else:
            self.wv = None
        self.embs = [np.zeros((300,)),
            np.random.random((300,))*0.01]
        self.word_index = {None:0, 'UNK':1}

    def _tokenize(self, string):
        if self.lower:
            example = string.strip().lower()
        else:
            example = string.strip()
        return re.findall(self.pattern, example)

    def fit(self, data):
        token_cnts = {}
        for ex in data:
            example_tokens = self._tokenize(ex)
            for token in example_tokens:
                if token not in token_cnts:
                    token_cnts[token] = 1
                else:
                    token_cnts[token] += 1

        index = 2
        for value, key in enumerate(token_cnts):
            if value < self.min_df:
                continue
            self.word_index[key] = index
            if self.wv is not None:
                if key in self.wv:
                    self.embs.append(self.wv[key])
                else:
                    self.embs.append(np.random.random((300,))*0.01)
            else:
                self.embs.append(np.random.random((300,))*0.01)
            index += 1

        self.embs = np.array(self.embs)
        del self.wv
        return

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        return_dataset = []
        for ex in data:
            example = self._tokenize(ex)
            index_example = []
            for token in example:
                if token in self.word_index:
                    index_example.append(self.word_index[token])
                else:
                    index_example.append(self.word_index['UNK'])
            return_dataset.append(index_example)

        return return_dataset

    def pad_data(self, data):
        max_len = np.max([len(x) for x in data])
        padded_dataset = []
        for example in data:
            zeros = [0]*(max_len-len(example))
            padded_dataset.append(example+zeros)
        return np.array(padded_dataset)
