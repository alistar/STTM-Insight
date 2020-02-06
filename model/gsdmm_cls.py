import sys
sys.path.append("../")
import pickle
import numpy as np
import pandas as pd
from gsdmm.gsdmm.mgp import *
from processing.text_processing import *
import random
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score

class GSDMM_Model:
    """
    inputs:
        corpus: the list of texts for topic modeling
        parameters for text manipulation using Corpus class: lowercase, remove_punct, strip_non_ascii,
            remove_digits, remove_stopwords, lemmatize
        parameters for MovieGroupProcess:
            n_class: number of classes GSDMS work with, should be larger than expected classes
            alpha: alpha parameter, corresponds to prior probability of instances choosing an empty class
            beta: beta parameter, corresponds to prior probability of instances joing a class w/o overlap
            n_iter: number of itterations

    attributes:
        input parameters for defining the model
        model: a GSDMS model defined by the input parameters
    methods:
        gsdmm_pred:
            outputs predicted classes per input text and the trained model (optional)
        hcv_score
            outputs h, c & v scores for a text and true labels
        inferences
            outputs n_doc_per_c, fractions, populars, freq_dists
    """

    def __init__(self, corpus=None, lowercase=True, remove_punct=True, strip_non_ascii=True,
                 remove_digits=True, remove_stopwords=True, lemmatize=True,
                 n_class=6, alpha=0.7, beta=0.6, n_iter=30, verbose=False):
        self.verbose = verbose
        self.n_class = n_class
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        if verbose:
            print(f"GSDMS model parameters: n_class = {n_class}, alpha = {alpha}, beta = {beta}, n_iters = {n_iters}.\n")
        self.model = MovieGroupProcess(K=n_class, alpha=alpha, beta=beta, n_iters=n_iter)

        text_cls = Corpus(corpus
                      , lowercase=lowercase
                      , remove_punct=remove_punct
                      , strip_non_ascii=strip_non_ascii
                      , remove_digits=remove_digits
                      , remove_stopwords=remove_stopwords
                      , lemmatize=lemmatize
                      , drop_empty_tokens=False
                      , verbose=True)
        self.text_tokenized = text_cls.tokenized
        self.nu_tokens = text_cls.nu_tokens
        self.model.fit(self.text_tokenized, self.nu_tokens)
        
    def predict(self, new_data=None):
        if new_data:
            return [self.model.choose_best_label(sent)[0] for sent in new_data]
        else:
            return [self.model.choose_best_label(sent)[0] for sent in self.text_tokenized]

    def hcv_score(self, new_data=None, true_labels=None):
        return homogeneity_completeness_v_measure(true_labels, self.predict(new_data=new_data))

    def inferences(self):
        model = self.model
        n_doc_per_c = np.sort(np.array(self.model.cluster_doc_count))
        n_doc_per_c = n_doc_per_c[::-1]
        fractions = (np.sort(np.array(self.model.cluster_doc_count)*100. / sum(self.model.cluster_doc_count)))
        fractions = fractions[::-1]
        populars = np.array(self.model.cluster_doc_count).argsort()[::-1]
        freq_dists = []
        for c in range(self.n_class):
            freq_dists.append(dict(sorted(model.cluster_word_distribution[populars[c]].items(),key=operator.itemgetter(1), reverse=True)))
        return n_doc_per_c, fractions, populars, freq_dists


if __name__ == "__main__":
    print(f"This code contains a GSDM class and its associated functions to work with GSDM topic modeling!")

    directory = '../data/'
    NYT_data = pd.read_csv(directory+'NYT_4topics_int.csv')
    NYT_titles = list(NYT_data.text)
    NYT_topics = list(NYT_data.topic)
    NYT_topics_int = list(NYT_data.topic_int)
    
    gsm = GSDMM_Model(corpus = NYT_titles, n_iter = 20)
    h,c,v = gsm.hcv_score(true_labels=NYT_topics_int)
    print(f"H: {h:.3f}, C: {c:.3f} & V_score: {v:.2f}")

    n_doc_per_c, fractions, populars, freq_dists = gsm.inferences()
    print(f"number of docs per cluster: {n_doc_per_c}")