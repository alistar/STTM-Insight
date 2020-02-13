import sys
sys.path.append("../")
import pickle
import numpy as np
import pandas as pd
from emb_kmeans_cls import Emb_Kmeans_Model
#from processing.text_processing import *
import random
##from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score


directory = '/Users/alistar/Projects/Insight_STTM/'
text_data = pd.read_csv(directory+'quora_questions.csv').sample(frac=0.1)
quora_questions = list(text_data.question_text)

with open(directory+'/Glove_quora_words.pkl', 'rb') as file2open:
    embedding_cls = pickle.load(file2open) 
emb_matrix = embedding_cls.matrix()
word2index = embedding_cls.word2index

GKmeans = Emb_Kmeans_Model(corpus=quora_questions, word2index=word2index, emb_matrix=emb_matrix, agg_func='mean',
                        n_class=40,max_iter=1000, max_no_improvement=50, random_state=4,
                        reassignment_ratio=0.0001, tol=1e-05, batch_size=1000,verbose=False)

    #h,c,v = GKmeans.hcv_score(true_labels=NYT_topics_int)
    #print(f"H: {h:.3f}, C: {c:.3f} & V_score: {v:.2f}")
    
n_doc_per_c, fractions, populars, freq_dists = GKmeans.inferences()
print(f"number of docs per cluster: {n_doc_per_c}")