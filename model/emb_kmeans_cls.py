import sys
sys.path.append("../")
import pickle
import numpy as np
import pandas as pd
from gsdmm.gsdmm.mgp import *
from processing.text_processing import *
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score

class Emb_Kmeans_Model:
    """
    inputs:
        corpus: the list of texts for topic modeling
        parameters for text manipulation using Corpus class: 
            lowercase, remove_punct, strip_non_ascii,
            remove_digits, remove_stopwords, lemmatize
            word2index, emb_matrix, agg_func
        parameters for MiniBatchKmeans:
            n_class, max_iter, max_no_improvement, random_state,
            reassignment_ratio, tol

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
                 word2index=None, emb_matrix=None, agg_func='mean', 
                 n_class=4, max_iter=1000, max_no_improvement=50, random_state=None,
                 reassignment_ratio=0.0001, tol=1e-05, batch_size=100, verbose=False):
        
        self.verbose = verbose
        self.n_class = n_class
        self.corpus = corpus

        self.model = MiniBatchKMeans(n_clusters=n_class, init='k-means++', max_iter=max_iter,
                                    batch_size=batch_size, verbose=0, compute_labels=True,
                                    random_state=random_state, tol=tol, max_no_improvement=max_no_improvement,
                                    init_size=None, reassignment_ratio=reassignment_ratio)
        text_cls = Corpus(self.corpus
                      , lowercase=lowercase
                      , remove_punct=remove_punct
                      , strip_non_ascii=strip_non_ascii
                      , remove_digits=remove_digits
                      , remove_stopwords=remove_stopwords
                      , lemmatize=lemmatize
                      , drop_empty_tokens=False
                      , verbose=True)
        
        self.text_tokenized = text_cls.tokenized
        self.lengths = text_cls.lengths
        
        self.text_emb = text_cls.agg_doc_rep(word2index=word2index,
                            emb_matrix=emb_matrix,
                            agg_func=agg_func)

        self.model.fit(self.text_emb)

    def predict(self, new_data=None):
        if new_data:
            return(self.model.predict(new_data))
        else:
            return(self.model.labels_)

    def hcv_score(self, new_data=None, true_labels=None):
        return homogeneity_completeness_v_measure(true_labels, self.predict(new_data=new_data))

    def inferences(self):
        df = pd.DataFrame(list(zip(self.corpus, self.text_tokenized, self.lengths, self.predict())), 
               columns =['Text', 'Tokenized', 'Lengths', 'Class'])
        cluster_doc_count = df.groupby(['Class'])['Class'].count().to_list()
        n_doc_per_c = np.sort(np.array(cluster_doc_count))
        n_doc_per_c = n_doc_per_c[::-1]
        fractions = (np.sort(np.array(cluster_doc_count)*100. / sum(cluster_doc_count)))
        fractions = fractions[::-1]
        populars = np.array(cluster_doc_count).argsort()[::-1]
        
        freq_dists = []
        for c in populars:
            freq_dist = dict(FreqDist([token for doc in df.Tokenized[df.Class == c].to_list() for token in doc]))
            freq_dists.append(dict(sorted(freq_dist.items(), key=operator.itemgetter(1), reverse=True)))


        return n_doc_per_c, fractions, populars, freq_dists

if __name__ == "__main__":
    print(f"This code contains Emb_Kmeans class and its associated functions to work with Sentence Embedding + K-means topic modeling!")

    directory = '../data/'
    NYT_data = pd.read_csv(directory+'NYT_4topics_int.csv')
    NYT_titles = list(NYT_data.text)
    NYT_topics = list(NYT_data.topic)
    NYT_topics_int = list(NYT_data.topic_int)
    
    try:
        with open('./Glove_quora_words.pkl', 'rb') as file2open:
            embedding_cls = pickle.load(file2open)
    except:
        from google_drive_downloader import GoogleDriveDownloader as gdd
        file_id = '1AgWnGC5pwS96u5zBjk_LfkAdDAuSkLcC'
        dest_path= './Glove_quora_words.pkl'
        gdd.download_file_from_google_drive(file_id=file_id,
                                            dest_path=dest_path,
                                            unzip=True)
        with open('./Glove_quora_words.pkl', 'rb') as file2open:
            embedding_cls = pickle.load(file2open)
        
    emb_matrix = embedding_cls.matrix()
    word2index = embedding_cls.word2index

    GKmeans = Emb_Kmeans_Model(corpus=NYT_titles, word2index=word2index, emb_matrix=emb_matrix, agg_func='mean',
                            n_class=4,max_iter=1000, max_no_improvement=50, random_state=4,
                            reassignment_ratio=0.0001, tol=1e-05, batch_size=100,verbose=False)

    h,c,v = GKmeans.hcv_score(true_labels=NYT_topics_int)
    print(f"H: {h:.3f}, C: {c:.3f} & V_score: {v:.2f}")
    
    n_doc_per_c, fractions, populars, freq_dists = GKmeans.inferences()
    print(f"number of docs per cluster: {n_doc_per_c}")
#    NYT_data['Glove-Kmeans-pred'] = GKmeans.predict()
#    NYT_data.to_csv('NYT_4topics_pred.csv', index=False)