
# Short Text Topic Modeling
Unsupervised topic extraction from a body of text is one of the most frequetn use cases in NLP. Commonly used methods for this task, such as TFIDF or LDA, rely on co-occurance of words that define different topics. Since this does not happen often in very short text, those methods fail to extract topics from short text such as article titles and social medai posts.

This repository contains code base required for building and applying two different models for topic extraction specifically desinged to perform well on short text. Moreover, it includes an app to interactively apply them to arbitrary collection of text.

# Models
The first approach is Gibbs Sampling for Drichlet Multinomial Mixture model, or in short GSDMM, is developed by [Yin & Wang 2014](dbgroup.cs.tsinghua.edu.cn). Compared to other available methods GSDMM has excellent performance while being computationally efficient (see [Qiang et al. 2019](https://arxiv.org/abs/1904.07695)). Topic modeling with this technique can be performed using the "GSDMM_model" class (see ./model/gsdmm_cls.py).

In the second approach I use transfer learning by combining sentence embedding and K-means clustering to find clusters of similar texts as proxy for the main topics. The current sentence embedding implementation uses agregated Glove word embedding which I found computationally efficient while achieveing BERT level representation performance. Topic modeling with this technique can be done using the "Emb_Kmeans_Model" class (see ./model/emb_kmeans_cls.py).

The two aforementioned models are integrated in a web app which allows the user to upload a collection of short text (as a csv file) and extract topics from it interactively. The user can perform topic extraction immediatly using the default settings. They can also opt for choosing between the two models, adjusting text processing, modifying model parameters or even fine tune how they want to see the results. You can see the app in action in the YouTube video below:

[![Alt text](https://img.youtube.com/vi/ckn0lQPvgFw/0.jpg)](https://www.youtube.com/watch?v=ckn0lQPvgFw)

# Intallation
With Git and Conda already installed and up2date you need to:

1- Clone this repo:
```
git clone https://github.com/alistar/STTM-Insight.git
git submodule init
git submodule update
```

2- Create a conda virtural environment called 'STTM-env' using the provided .yml file. From within the STTM-Insight directory this can be done by:
```
conda env create -f configs/config.yml
conda activate STTM-env
```
# Topic extraction using the interactive App
First you need to launch the web-app locally in the "STTM-env" environment by:
```
streamlit run model/sttm_streamlit.py
```

After the last command you should be able to connect to the web-app and interact with it. The app should pop-up in your internet browser but if not, you can manually got to the given local URL through "http://localhost:8509".

You can run a demo by pressing the big bottom in the middle of the page "Extract 4 topics using Glove+K-means method". This will load a test data-set which contains 3k article titles chosen from 4 different New York Times sections, perform sentence embedding using Glove and finally finding 4 clusters using K-means clustring. The the WordCloud of the 4 clusters/topics will be shown along with their top 10 most frequent words.

_Note that the first time you are running the app and using Glove embedding method, it needs to download a file containing Glove embeddings. The file is 274M and depending on the internet speed might take some time. This only happens needs to be done once and subsequent runs will be faster_

Besides using the default settings, the web-app has many features enabling the user to adjust the model and its input according to their needs. They can upload an arbitrary body of text as .csv file, choose the GSDMM topic extraction model, change the text processing steps, adjust various model parameters and the number of clusters in the output.


***This is a project I developed during my first 3 weeks at Insight as an AI fellow***