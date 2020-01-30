import sys
sys.path.append("../")
import pickle
import numpy as np
import pandas as pd
from gsdmm.gsdmm.mgp import *
from processing.text_processing import *
import random
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score

def gsdmm_test(data, labels, K=6, alpha=0.3, beta=0.2, n_iters=10):
    """ runs a labeled text set through GSDMM pipeline and returns hcv (Homogeneity, Completeness, V_measure) scores and model
        inputs: data as Corpus class, labels as list of topics 
    """
    
    mgp = MovieGroupProcess(K=K, alpha=alpha, beta=beta, n_iters=n_iters)
    y = mgp.fit(data.tokenized, data.nu_tokens)
    pred_classes = [mgp.choose_best_label(sent)[0] for sent in data.tokenized]
    h, c, v = homogeneity_completeness_v_measure(labels, pred_classes)
    return h, c, v, mgp

def gsdmm_pred(data, K=6, alpha=0.3, beta=0.2, n_iters=10):
    """ runs a text set through GSDMM pipeline and returns hcv (Homogeneity, Completeness, V_measure) scores and model """
    mgp = MovieGroupProcess(K=K, alpha=alpha, beta=beta, n_iters=n_iters)
    y = mgp.fit(data.tokenized, data.nu_tokens)
    pred_classes = [mgp.choose_best_label(sent)[0] for sent in data.tokenized]
    return pred_classes, mgp

def gsdm_insight(gsdm_model, nc2show, nword2show):
    n_doc_perc = np.sort(np.array(gsdm_model.cluster_doc_count))
    print(f"number of documetns per class: {n_doc_perc[::-1]}")
    fractions = (np.sort(np.array(gsdm_model.cluster_doc_count)*100. / sum(gsdm_model.cluster_doc_count)))
    populars = np.array(gsdm_model.cluster_doc_count).argsort()[::-1]
    np.set_printoptions(precision=2)
    print(f"% of documents per class:{fractions[::-1]}")
    print(f"most populus classes: {populars}")

    for i in range(nc2show):
        try:
            print(f"Cluster no {i} top {nword2show} words are: {dict(sorted(gsdm_model.cluster_word_distribution[populars[i]].items(),key=operator.itemgetter(1), reverse=True)[0:nword2show])}\n\n")
        except IndexError:
            print(f"no more clusters/word are present!")
    return fractions, populars

if __name__ == "__main__":
    print(f"This code contains functions to work with GSDM topic modeling!")
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

    h,c,v, gsdm_model = gsdmm_test(NYT_text_cls, NYT_topics_int)
    print(f"H:{h}, C:{c}, V:{v}\n\n")
    _, _ = gsdm_insight(gsdm_model, 4, 5)
    