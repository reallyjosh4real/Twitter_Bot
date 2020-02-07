# The Natural Process of Language Processing 

## Table of Contents

## Introduction
72% of U.S smartphone users use less than 7 apps in a day. In response to this consumer behavior, the social platforms that we all know and love have started letting companies build chatbots with them.

 The chatbot can talk to you through different channels, such as Facebook Messenger, Siri, WeChat, Twitter, SMS, Slack, Skype, among others. 23.5 trillion WhatsApp messages sent every year. Twitter also holds a huge influence in the electronic messaging space. For example, there were 103 million tweets sent about the Winter Olympic Games and, in total, this led to 33.6 billion Impressions (views on and off Twitter) of tweets.

So first I wanted to explore if I would be able to predict a response to a general question given the dataset.
## Data 
I decided to work with a twitter customer question and response data set that included a total of 2,811,774 tweets and was directed at 108 different companies. The dataset had 7 columns. 

<code>tweet_id  author_id	inbound  created_at text response_tweet_id in_response_to_tweet_id</code>

<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-06 at 10.47.21 PM.PNG">


## EDA
I merged question tweets with response tweets on author_id and response_tweet_id. Getting the paired QA data was in my mind the first step the being able to predict answers given questions. 

<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-06 at 2.32.16 PM.PNG">


<img src='/Users/ramozo_88/Twitter_Bot/images/tweet_counts_per_company.PNG'>

From here I was interested in the emojis in the text. I was able to run the text through a spacy pipeline with an emoji paser. I managed to get the most frequent emojis used by customers directed at companies. Mostly emojis themed in frustration.

<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-06 at 2.31.21 PM.PNG">

I then used scattertext with a spacy pipline to plot a spatial word frequency plot with blue representing Apple Support and the red is Amazon Help. This graph basically shows that there are alot of similar words used by both groups represented by the light yellow are in the middle of the plot.  The top left corner represents most frequent words used specifically by Apple Support while the bottom right plotted words are representative of frequent words characteristic to Amazon Help.

<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-06 at 2.34.28 PM.PNG">

There is a package called empath that once imported with scattertext can group words into topics.

<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-06 at 2.36.07 PM.PNG">


## Model

### K-means / Hard Clustering

<img src="/Users/ramozo_88/Twitter_Bot/images/kmeans_elbow_graph.PNG">

Distortion: mean sum of squared distances to centers
Customer Questions:

<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-07 at 3.59.25 AM.png">
Company Responses:

<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-07 at 4.03.00 AM.png">

### LDA / Soft Clustering 
Customer Questions:

<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-07 at 2.39.42 AM.png">
<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-07 at 2.40.08 AM.png">

Company Responses:

<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-07 at 2.44.15 AM.png">
<img src="/Users/ramozo_88/Twitter_Bot/images/Screen Shot 2020-02-07 at 2.44.32 AM.png">


## Future Work 

I would like to apply RNN and Seq2Seq to my X and y text to hopefully get a more intuitive way to predict responses given a question. 


