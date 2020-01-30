import sys
sys.path.append("../")
import pickle
import numpy as np
import pandas as pd
from processing.text_processing import *
import random
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

def embedding_kmeans_test(data, labels, embedding_cls,
                          do_minmax=True,
                          agg_func='mean',
                          n_clusters=4,
                          init='k-means++', max_iter=1000,
                          batch_size=10 * 5, verbose=0, compute_labels=True,
                          random_state=None, tol=0.0001, max_no_improvement=10,
                          init_size=None, reassignment_ratio=0.01):
    """ use Glove for sentence embeding of a set of labeled text and
        run it through K-means pipeline and returns hcv (Homogeneity, Completeness, V_measure) scores 
    """
    Kmeans_batch = MiniBatchKMeans(n_clusters=n_clusters, init=init, max_iter=max_iter,
                                    batch_size=batch_size, verbose=verbose, compute_labels=compute_labels,
                                    random_state=random_state, tol=tol, max_no_improvement=max_no_improvement,
                                    init_size=init_size, reassignment_ratio=reassignment_ratio)
    
    emb_matrix = embedding_cls.matrix()
    
    sent_emb = data.agg_doc_rep(word2index=embedding_cls.word2index,
                                emb_matrix=emb_matrix,
                                agg_func=agg_func)
    if do_minmax:
        emb_scaler = MinMaxScaler()
        emb_norm = emb_scaler.fit_transform(sent_emb)
        sent_emb = emb_norm
        
    Kmeans_batch.fit(sent_emb)
    if compute_labels:
        pred_classes = Kmeans_batch.labels_
    else:
        pred_classes = Kmeans_batch.predict(sent_emb)
    h, c, v = homogeneity_completeness_v_measure(labels, pred_classes)
    return h, c, v, Kmeans_batch

def embedding_kmeans_pred(data, labels, embedding_cls,
                          do_minmax=True,
                          agg_func='mean',
                          n_clusters=4,
                          init='k-means++', max_iter=1000,
                          batch_size=10 * 5, verbose=0, compute_labels=True,
                          random_state=None, tol=0.0001, max_no_improvement=10,
                          init_size=None, reassignment_ratio=0.01):
    """ use Glove for sentence embeding of a set of labeled text and
        run it through K-means pipeline and returns hcv (Homogeneity, Completeness, V_measure) scores 
    """
    Kmeans_batch = MiniBatchKMeans(n_clusters=n_clusters, init=init, max_iter=max_iter,
                                    batch_size=batch_size, verbose=verbose, compute_labels=compute_labels,
                                    random_state=random_state, tol=tol, max_no_improvement=max_no_improvement,
                                    init_size=init_size, reassignment_ratio=reassignment_ratio)
    
    emb_matrix = embedding_cls.matrix()
    
    sent_emb = data.agg_doc_rep(word2index=embedding_cls.word2index,
                                emb_matrix=emb_matrix,
                                agg_func=agg_func)
    if do_minmax:
        emb_scaler = MinMaxScaler()
        emb_norm = emb_scaler.fit_transform(sent_emb)
        sent_emb = emb_norm
        
    Kmeans_batch.fit(sent_emb)
    if compute_labels:
        pred_classes = Kmeans_batch.labels_
    else:
        pred_classes = Kmeans_batch.predict(sent_emb)
    return pred_classes, Kmeans_batch

if __name__ == "__main__":
    print(f"This code contains functions to work with sentence embedding + Kmeans clustring to extract topics")
    print(f"Below we run a demo using NYT article titles from 4 different topics")
    directory = '../data/'
    try:
        NYT_data = pd.read_csv(directory+'NYT_4topics_int.csv')
    except:
        import s3fs
        NYT_data = pd.read_csv('s3://alistar100/Insight/Data/NYT_data/NYT_4topics_int.csv')
    NYT_titles = list(NYT_data.text)
    NYT_topics = list(NYT_data.topic)
    NYT_topics_int = list(NYT_data.topic_int)

    NYT_text_cls = Corpus(NYT_titles
                      , lowercase=True
                      , remove_punct=True
                      , only_alphanum=False
                      , strip_non_ascii=True
                      , remove_digits=True
                      , remove_stopwords=True
                      , lemmatize=True
                      , verbose=True)

    glove_file = "/Users/alistar/Projects/Kaggel/quora-insincere-que/embeddings/glove.840B.300d/glove.840B.300d.txt"
    embedding_cls = Embedding_cls(load_glove(file=glove_file))

    h,c,v, embd_kmeans_model = embedding_kmeans_test(NYT_text_cls, NYT_topics_int, embedding_cls,
                                               do_minmax=True,
                                               agg_func='mean',
                                               n_clusters=4)
    print(f"H:{h}, C:{c}, V:{v}\n\n")
    