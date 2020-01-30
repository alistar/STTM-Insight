import sys
sys.path.append("../")
import pickle
import numpy as np
import pandas as pd
from processing.text_processing import *
import random
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def lda_test(data, labels, n_components=4, max_df = 0.5, min_df = 10, random_state=0, freq_mode='TF'):
    """ runs a labeled text set through LDA pipeline and returns hcv (Homogeneity, Completeness, V_measure) scores 
        inputs: 'data' as Corpus class, labels as list of topics
    """
    lda_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                    stop_words = 'english',
                                    lowercase = True,
                                    token_pattern = r'\b[a-zA-Z]{3,}\b',
                                    max_df = max_df, 
                                    min_df = min_df)
    if freq_mode == 'TFIDF':
        print("using TFIDF vectorization!")
        lda_vectorizer = TfidfVectorizer(**lda_vectorizer.get_params())
    else:
        print("using TF vectorization!")
        
    dtm_freq = lda_vectorizer.fit_transform(list(data.untokenize())) 
        
    lda_model = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    lda_model.fit(dtm_freq)


    lda_topics = lda_model.transform(dtm_freq)#(dtm_tfifd)
    pred_classes = [topics.argmax() for topics in lda_topics]

    h, c, v = homogeneity_completeness_v_measure(labels, pred_classes)
    return h, c, v, lda_model

def lda_pred(data, n_components=4, max_df = 0.5, min_df = 10, random_state=0, freq_mode='TF', visualize=False):
    """ runs a text set through LDA pipeline and returns topics 
        inputs: 'data' as Corpus class, labels as list of topics
    """
    lda_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                    stop_words = 'english',
                                    lowercase = True,
                                    token_pattern = r'\b[a-zA-Z]{3,}\b',
                                    max_df = max_df, 
                                    min_df = min_df)
    if freq_mode == 'TFIDF':
        print("using TFIDF vectorization!")
        lda_vectorizer = TfidfVectorizer(**lda_vectorizer.get_params())
    else:
        print("using TF vectorization!")
        
    dtm_freq = lda_vectorizer.fit_transform(list(data.untokenize())) 
        
    lda_model = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    lda_model.fit(dtm_freq)


    lda_topics = lda_model.transform(dtm_freq)#(dtm_tfifd)
    pred_classes = [topics.argmax() for topics in lda_topics]
    
    return pred_classes, lda_model, dtm_freq, lda_vectorizer



if __name__ == "__main__":
    print(f"This code contains functions to work with LDA topic modeling!")
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

    h, c, v, _ = lda_test(NYT_text_cls, NYT_topics_int, n_components=4, freq_mode='TFIDF')
    print(f"H:{h}, C:{c}, V:{v}\n\n")
    
    _, lda_model, dtm_freq, lda_vectorizer = lda_pred(NYT_text_cls, n_components=4, freq_mode='TF', visualize=True)
#    try:
    import pyLDAvis
#   except:
# Install a pip package in the current kernel
#        import sys
#        {sys.executable} -m pip install pyLDAvis
#    import pyLDAvis
    import pyLDAvis.sklearn
 #   pyLDAvis.enable_notebook()
    pyLDAvis.sklearn.prepare(lda_model, dtm_freq, lda_vectorizer)
    