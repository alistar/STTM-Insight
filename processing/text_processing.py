import os
#import wget
import numpy as np
import operator
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk.corpus import stopwords
#from tqdm import tqdm_notebook as tqdm
import string
import re

#try:
#    stopwords = stopwords.words('english')
#except:
from nltk import download
#download('stopwords')
"""
Classes:
"""


class Embedding_cls:
    """
    inputs:
        embedding_dic: a dictionary containing words and their representations
        vocabs2include=None: a list of words to include in the embedding
    attributes:
        embedding_dic: copy of the input dictionary
        n_vocab: length of vocabulary
        emb_dim: embedding dimensionality
        word2index: a dictionary containing {word:index}
        index2word: a dictionary containing {index:word}
    methods:
        matrix:
            outputs embedding matrix np.array(n_vocab+1, dim_emb)
    """

    def __init__(self, embedding_dic, vocabs2include=None, verbose=1):
        self.verbose = verbose
        if vocabs2include is not None:
            embedding_dic_new = {}
            for key in vocabs2include:
                try:
                    embedding_dic_new[key] = embedding_dic[key]
                except KeyError:
                    pass
#                    print(f"vocab {key} not found in embedding")
            embedding_dic = embedding_dic_new
        self.embedding_dic = embedding_dic
        self.n_vocab = len(embedding_dic)
        self.emb_dim = len(embedding_dic[list(embedding_dic.keys())[0]])
        if verbose:
            print(f"Embedding is loaded with {len(embedding_dic)} words each having {len(embedding_dic[list(embedding_dic.keys())[0]])} dimensions")
        vocabs = [word for word in embedding_dic.keys()]
        indices = range(len(embedding_dic))
        self.word2index = {vocabs[i]: i for i in indices}
        self.index2word = {i: vocabs[i] for i in indices}

    def matrix(self):
        embedding_matrix = np.zeros([self.n_vocab + 2, self.emb_dim])
        # Note that a last zero-vector is assigned for the OOV representation
        # There is also a second zero-vector set aside for buffering
        for index, word in self.index2word.items():
            embedding_matrix[index, :] = self.embedding_dic[word]
        return embedding_matrix

    def load_glove(file="./glove.840B.300d.txt"):
        """
        inputs:
            file: the address of file containing embedding info
        outputs:
            embeddings_index: a dictionary containing {embedding words: their representation}
        """

        def get_coefs(word, *arg):
            return word, np.asarray(arg, dtype='float32')

        try:
            open(file).close()
        except OSError:
            url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
            adrs = "./glove.840B.300d"
            print(f"Embedding file {file} unreachable")
            print(f"downloading {url} int {adrs}")
#        wget.download(url, adrs+".zip")
            os.system("wget "+url)
            os.system("unzip "+adrs+".zip")
            os.system("ls")
            file = adrs+".txt"

        embeddings_dict = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        return embeddings_dict


class Corpus:
    """
    inputs:
        docs: a list containing different text documents
        lowercase (default to False): if True, all words be lower-cased at the beginning
        remove_punct(default to False): if True removes punctuation signs
        strip_non_ascii(default to False): if True all non-asci chars are removed
        tokenizer (optional): tokenizer to replace the default "NLTK.TreeBankWordTokenizer"
        remove_digits (default to False): if True all digits will be removed
        remove_stopwords (default to False): if True NLTK."English" stopwords will be removed
        lemmatize (default to False: if True, NLTK.WordNetLemmatizer will be used to Lemmatize
        drop_empty_tokens (default to True): if False, docs which are empty after cleaning will be tokenized w/o any cleaning

    attributes:
        original: orignal documents
        tokenized: a list containing token lists for each document
        lengths: a list containing the length (tokens) of each documents
        token_dist: a dictionary containing {token: frequency_in_corpus)
        n_docs: number of docs in the corpus
        n_tokens: total number of tokens in the corpus
        nu_tokens: total number of unique tokens in the corpus
    methods:
        to_indices:
            inputs:
                word2index: a dict of {token:emb_index}
                oov_ind (optional): to replace the default max(emb_index)+1
                max_length (optional): if defined as int it will set the length of each docs
                buffer_ind (optional): if defined, it will be used for buffer_ind in conjunction with max_length
                    if not defined and in case of "max_length > 0", it will be set to max(word2index_ind)+2
            output:
                a 2d list (doc,ind) containing indices instead of words
        agg_doc_rep:
            inputs:
                word2index: a dict of {token:emb_index}
                emb_matrix: embedding matrix np.array(emb_index, dim_emb)
                oov_ind (optional): to replace the default index which is max(emb_index)+1
                agg_func: the aggregate function to be applied (mean,median,std,min,max)
            output:
                a 2d list (doc,agg_rep) containing aggregate representation of each document
        oov_stats:
            inputs
                word2index: a dict of {token:emb_index}
            output:
                a dictionary containing {oov token: frequency_in_corpus}
    """

    def __init__(self, docs, lowercase=False, remove_punct=False, strip_non_ascii=False, tokenizer=TreebankWordTokenizer(), remove_digits=False,remove_stopwords=False, lemmatize=False, drop_empty_tokens=True, verbose=1):
        self.verbose = verbose
        doc_copy = docs.copy()
        if lowercase:
            docs = [doc.lower() for doc in docs]
        if remove_punct:
            translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            docs = [doc.translate(translator) for doc in docs]
        if strip_non_ascii:
            def remove_non_ascii(string):
                stripped = (c for c in string if 0 < ord(c) < 127)
                return ''.join(stripped)
            docs = [remove_non_ascii(doc) for doc in docs]
        tokenized_docs = [tokenizer.tokenize(doc) for doc in docs]
        if remove_digits:
            tokenized_clean = []
            for doc in tokenized_docs:
                tokenized_clean.append([token for token in doc if not token.isdigit()])
            tokenized_docs = tokenized_clean
        if remove_stopwords:
            stopwordslist = list(stopwords.words('english'))
            stopwordslist= stopwordslist + [s.capitalize() for s in stopwordslist]
            tokenized_clean = []
            for doc in tokenized_docs:
                tokenized_clean.append([token for token in doc if token not in stopwordslist])
            tokenized_docs = tokenized_clean
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            tokenized_clean = []
            for doc in tokenized_docs:
                tokenized_clean.append([lemmatizer.lemmatize(token) for token in doc])
            tokenized_docs = tokenized_clean
        #Below checks empty tokenized docs that could happend after e.g., stopword removal and reverts them to original
        for i, doc in enumerate(tokenized_docs):
            if len(doc) < 1:
                if verbose:
                    print(i, doc_copy[i])
                if drop_empty_tokens:
                    doc_copy.pop(i)    
                    tokenized_docs.pop(i)
                else: 
                    tokenized_docs[i] = tokenizer.tokenize(doc_copy[i])
        self.original = doc_copy
        self.tokenized = tokenized_docs
        self.lengths = [len(doc) for doc in tokenized_docs]
        freq_dist = dict(FreqDist([token for doc in tokenized_docs for token in doc]))
        self.token_dist = dict(sorted(freq_dist.items(), key=operator.itemgetter(1), reverse=True))
        self.n_docs = len(docs)
        self.n_tokens = np.array([val for val in freq_dist.values()]).sum()
        self.nu_tokens = len(freq_dist)

        if verbose:
            print(
                f"The corpus contains {np.array([val for val in freq_dist.values()]).sum()} total tokens where {len(freq_dist)} are unique")

    def untokenize(self):
        docs = []
        for doc in self.tokenized:
            docs.append(' '.join(doc).strip())
        return docs
        

    def to_indices(self, word2index=None, oov_ind=None, max_length=None, buffer_ind=None):
        if not isinstance(word2index, dict):
            raise TypeError("'word2index' should be a 'dict' but its not defined properly!")
        if not isinstance(oov_ind, int):
            if self.verbose:
                print(f"'oov_ind' is not provided as an integer key. Its set to max(ind)+1.")
            oov_ind = max(list(word2index.values())) + 1
        docs2indices = []
        for doc in self.tokenized:
            doc2indices = []
            for token in doc:
                try:
                    doc2indices.append(word2index[token])
                except KeyError:
                    doc2indices.append(oov_ind)
            docs2indices.append(doc2indices)
        if (isinstance(max_length, int)) and (max_length > 0):
            if not isinstance(buffer_ind, int):
                buffer_ind = max(list(word2index.values())) + 2
            docs2indices_fixed = []
            for doc in docs2indices:
                doc2indices_fixed = []
                for i in range(max_length):
                    try:
                        doc2indices_fixed.append(doc[i])
                    except IndexError:
                        doc2indices_fixed.append(buffer_ind)
                docs2indices_fixed.append(doc2indices_fixed)
            docs2indices = docs2indices_fixed
        return docs2indices

    def agg_doc_rep(self, word2index=None, emb_matrix=None, oov_ind=None, agg_func='mean'):
        if not isinstance(word2index, dict):
            raise TypeError("'word2index' should be a 'dict' but its not defined properly!")
        if (emb_matrix.shape[0] - len(word2index) > 2) or (emb_matrix.shape[0] < len(word2index)):
#            print(f"'emb_matrix.shape': {emb_matrix.shape}, len(word2index): {len(word2index)}")
            raise TypeError("'emb_matrix' should be an numpy array with dimension ({len(word2index)}(+2), emb_dim)!")
        if not isinstance(oov_ind, int):
            if self.verbose:
                print(f"'oov_ind' is not provided as an integer key. Its set to max(ind)+1.")
            oov_ind = max(list(word2index.values())) + 1
        agg_funcs = ('mean', 'median', 'std', 'min', 'max')
        if agg_func not in agg_funcs:
            raise TypeError("The provided 'agg_func':{agg_func} is not among available functions {agg_funcs}")

        docs2indices = self.to_indices(word2index=word2index, oov_ind=oov_ind)

        def agg_1doc_rep(doc, emb_matrix=emb_matrix, agg_func=agg_func):
            doc_emb = [emb_matrix[ind][:] for ind in doc]
            return eval('np.' + agg_func + '(doc_emb, axis=0)')

        return [agg_1doc_rep(doc, emb_matrix=emb_matrix, agg_func=agg_func) for doc in docs2indices]

    def agg_doc_multi_reps(self, word2index=None, emb_matrix=None, oov_ind=None, agg_funcs=['min','mean','max']):
        if self.verbose:
            print(f"concatenating multiple aggregate representation of each document: {agg_funcs}")
        doc_reps = self.agg_doc_rep(word2index=word2index, emb_matrix=emb_matrix,
                                    oov_ind=oov_ind, agg_func=agg_funcs[0])
        for agg_func in agg_funcs[1:]:
            doc_reps = np.concatenate((doc_reps,
                                       self.agg_doc_rep(word2index=word2index, emb_matrix=emb_matrix,
                                                        oov_ind=oov_ind, agg_func=agg_func)),
                                      axis=1)
        return doc_reps


    def oov_stats(self, word2index=None):
        if not isinstance(word2index, dict):
            raise TypeError("'word2index' should be a 'dict' but its not defined properly!")
        outliers = {token: freq for token, freq in self.token_dist.items() if token not in word2index.keys()}
        outliers = dict(sorted(outliers.items(), key=operator.itemgetter(1), reverse=True))
        if self.verbose:
            print(
                f"The provided 'word2index' fails to cover {len(outliers) / self.nu_tokens:{2}.{3}} of unique tokens in the corpus")
            print(
                f"This means {sum([freq for token, freq in outliers.items()]) / self.n_tokens:{2}.{3}} of all tokens in the corpus are not covered!")
        return outliers


if __name__ == "__main__":
    print(f"This code contains some useful Classes (Embedding, Corpus) and functions (load_glove) for text processing")
