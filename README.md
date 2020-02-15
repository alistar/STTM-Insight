# Short Text Topic Modeling

Unsupervised topic extraction from a body of text is one of the most frequetn use cases in NLP. Commonly used methods for this task, such as TFIDF or LDA, rely on co-occurance of words that define different topics. Since this does not happen often in very short text, those methods fail to extract topics from short text such as article titles and social medai posts.

This repository contains code base required for building and applying two different models for topic extraction specifically desinged to perform well on short text. Moreover, it includes an app to interactively apply them to arbitrary collection of text.

# Models

The first approach is Gibbs Sampling for Drichlet Multinomial Mixture model, or in short GSDMM developed by [Yin & Wang 2014](dbgroup.cs.tsinghua.edu.cn). In recent years several other methods have been proposed for short text topic extraction but GSDMM is proven to have excellent performance while being computationally efficient as shown by [Qiang et al. 2019](arxiv:1904.07695). Topic modeling using this technique can be performed using the "GSDMM_model" class (see model/gsdmm_cls.py).

In the second approach I use transfer learning by combining sentence embedding and K-means clustering to find clusters of similar texts as proxy for the main topics. The current sentence embedding implementation uses agregated Glove word embedding which I found computationally efficient while achieveing BERT level representation performance. Topic modeling using this technique can be done using the "Emb_Kmeans_Model" class (see model/emb_kmeans_cls.py).

# App

The two aforementioned models are integrated in a Streamlit app which allows the user to upload a collection of short text (as a csv file) and extract topics from it interactively. The user can perform topic extraction immediatly using the default settings. They can also opt for choosing between the two models, adjusting text processing, modifying model parameters or even fine tune how they want to see the results.