import pandas as pd
import numpy as np
import pandas as pd
import re
import nltk
from collections import Counter
import spacy
from spacy_langdetect import LanguageDetector
import string
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import scattertext as st
from scattertext import word_similarity_explorer
from IPython.display import IFrame
from pprint import pprint
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from yellowbrick.cluster import KElbowVisualizer
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()
pd.options.display.max_rows = 1000
pd.options.mode.chained_assignment = None

punctuations = string.punctuation
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
stop_words = stop_words.union(set(apple_stop_words))

parser = English()

def spacy_tokenizer(sentence):
   
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens

def clean_data():
    tweets = pd.read_csv('/Users/ramozo_88/Pot_Bot/data/customer-support-on-twitter/twcs/twcs.csv')
    first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]
    df_question_response = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                  right_on='in_response_to_tweet_id')
    df_question_response = df_question_response[df_question_response.inbound_y ^ True]
    df_question_response = df_question_response[["author_id_x","created_at_x","text_x","author_id_y","created_at_y","text_y"]]
    df_apple_question_response = df_question_response[df_question_response["author_id_y"]=="AppleSupport"]
    return df_apple_question_response

def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))


if __name__ == "__main__":

    df_apple_question_response = clean_data()
    apple_stop_words = ['applesupport', '115858', 'apple', 'https', 'gdrqu22ypt', 
                    'xxaxeesrt9', 'ibiy3vmgpj', '80yrnjdfdk', 'qodbosp4wz',
                   '116333', 'co', 've', '11', 'gdrqu2kzhr', 'etpvyvfyd8', '08olhchdnv'
                   '0abquuca1w', '10ypnthryf', '0jgzopoxcv', '1inv8mjbuc', '1j584kcilt', 
                    '1hqerbwkjv', '0pqh8fn3nu']
    bow_vector = CountVectorizer(max_features=1000, tokenizer = spacy_tokenizer, ngram_range=(1,1))
    tfidf_vector = TfidfVectorizer(max_features=1000, tokenizer = spacy_tokenizer)
    X = df_apple_question_response['text_x'] 
    ylabels = df_apple_question_response['text_y'] 

    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)
    
    num_features = 1000
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words=stop_words)
    tf = tf_vectorizer.fit_transform(X_train)
    num_topics = 13
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online',random_state=0, n_jobs=-1)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    num_top_words = 15
    display_topics(lda, tf_feature_names, num_top_words)

    print("Model perplexity: {0:0.3f}".format(lda.perplexity(tf)))

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(X_train)
    true_k = 13
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    top_centroids = model.cluster_centers_.argsort()[:,-1:-11:-1]
    terms = vectorizer.get_feature_names()
    print('Top features (words) for each cluster:')
    for num, centroid in enumerate(top_centroids):
        print("Cluster%d: %s" % (num, ", ".join(terms[i] for i in centroid)))

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(X_train)
    model = KMeans(init='k-means++', max_iter=100, n_init=1)
    visualizer = KElbowVisualizer(model, k=(3,15), timings=False, locate_elbow=False )

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show(outpath='/Users/ramozo_88/Twitter_Bot/images/kmeans_elbow_graph.PNG') 