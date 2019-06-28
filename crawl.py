
Created on Fri May 11 15:42:43 2018

import re
import json
import numpy as np
from collections import Counter
from TwitterAPI import TwitterAPI # in case you need to install this package, see practical 6
#!pip install TwitterAPI
from sklearn.cluster import KMeans

import requests

# disabling urllib3 warnings
requests.packages.urllib3.disable_warnings()

import matplotlib.pyplot as plt
#%matplotlib inline

'''
If you need add any additional packages, then add them below
'''
import os # Isaac is running this assignment on an OSX system 

keywords = ["gift", "science", "actor"]

# Twitter API credentials 
CONSUMER_KEY = "bTUFlq4Eg3VX2X56c36ywQNyN"
CONSUMER_SECRET = "xc80ipckQarVowccnYziWiUPuxSneo426kvFPIxIz38X9rhvx7"
OAUTH_TOKEN = "986025247408205825-E04X5igF18TuJYTqlhS6WQEKf1aoMm3"
OAUTH_TOKEN_SECRET = "65bBCzLu3epG8ifJyTorhArp28pT2zOrOZGB0k7t3IPHe"


# Authonticating with your application credentials
api = TwitterAPI(
                 CONSUMER_KEY,
                 CONSUMER_SECRET,
                 OAUTH_TOKEN,
                 OAUTH_TOKEN_SECRET
                ) #authenticating with the TwitterAPI

# geo coordinations of the desired place
AUS_LAT = -24.396176
AUS_LONG = 133.247614
AUS_RADIUS = 1500

def retrieve_tweets(api, keyword, batch_count, total_count):
    """
    collects tweets using the Twitter search API
    
    api:         Twitter API instance
    keyword:     search keyword
    batch_count: maximum number of tweets to collect per each request
    total_count: maximum number of tweets in total
    """
    
    # the collection of tweets to be returned
    tweets = [] #empty list
    
    # the number of tweets within a single query
    batch_count = str(batch_count)
    
    '''
    You are required to insert your own code where instructed to perform the first query to Twitter API.
    Hint: revise the practical session on Twitter API on how to perform query to Twitter API.
    '''
    # per the first query, to obtain max_id_str which will be used later to query sub
    resp = api.request('search/tweets', {'q':keyword,
                                         'count':batch_count,
                                         'lang':'en',
                                         'result_type':'recent',
                                         'geocode':'{},{},{}mi'.format(AUS_LAT, AUS_LONG, AUS_RADIUS)
                                        } 
                      )
    
    # store the tweets in a list
    tweets += resp.json()['statuses']
    
    # find the max_id_str for the next batch
    ids = [tweet['id'] for tweet in tweets]
    max_id_str = str(min(ids))

    # loop until as many tweets as total_count is collected
    number_of_tweets = len(tweets)
    
    while number_of_tweets < total_count:
        
        print("{} tweets are collected for keyword {}. Last tweet created at {}".format(number_of_tweets, 
                                                                                        keyword, 
                                                                                        tweets[number_of_tweets-1]['created_at']))
        
        resp = api.request('search/tweets', {'q':keyword, 
                                             'count':batch_count,
                                             'lang':'en',
                                             'result_type':'recent',
                                             'max_id':max_id_str,
                                             'geocode':'{},{},{}mi'.format(AUS_LAT, AUS_LONG, AUS_RADIUS)
                                            }
                          )

        tweets += resp.json()['statuses']
        ids = [tweet['id'] for tweet in tweets]
        max_id_str = str(min(ids))
        number_of_tweets = len(tweets)
        
    print("{} tweets are collected for keyword {}. Last tweet created at {}".format(number_of_tweets, 
                                                                                    keyword, 
                                                                                    tweets[number_of_tweets-1]['created_at']))
    
    
    return tweets[:total_count]

k1_tweets = retrieve_tweets(api,keywords[0], 50, 100) #calling the previously defined function to interact with the api                     
k2_tweets = retrieve_tweets(api, keywords[1], 50, 100)
k3_tweets = retrieve_tweets(api,keywords[2], 50, 100)

print("'gift' tweet count: {}".format(len(k1_tweets))) #iterating through the length of each keyword array and printing the result
print("'science' tweet count: {}".format(len(k2_tweets)))
print("'actor' tweet count: {}".format(len(k3_tweets)))

print("Type of the first tweet in k1_tweets: {}".format(type(k1_tweets[0])))

print("Fields of first tweet of k1_tweets: {}".format(k1_tweets[0].keys())) #printing out the dict keys


print("\nThe text of the first tweet for \"{}\":\n".format(keywords[0]))
print(k1_tweets[0]['text'])

print('\nThe text of the first tweet for \"{}\":\n'.format(keywords[1]))
print(k2_tweets[0]['text'])

print('\nThe text of the first tweet for \"{}\":\n'.format(keywords[2]))
print(k3_tweets[0]['text'])


def is_short_tweet(tweet):
    '''
    Check if the text of "tweet" has less than 50 characters
    '''
    return len(tweet['text']) < 50

k1_tweets_filtered = list(filter(lambda x: not is_short_tweet(x), k1_tweets)) #calling our previously function and placing all objects which are greater than 50 characters into a new "filtered array"
k2_tweets_filtered = list(filter(lambda x: not is_short_tweet(x), k2_tweets))
k3_tweets_filtered = list(filter(lambda x: not is_short_tweet(x), k3_tweets))

# these lines below print the number of tweets for each keyword before and after filtered.
k1_len, k2_len, k3_len = len(k1_tweets), len(k2_tweets), len(k3_tweets)
k1_f_len, k2_f_len, k3_f_len = len(k1_tweets_filtered), len(k2_tweets_filtered), len(k3_tweets_filtered)

print(k1_len, k1_f_len) #printing length of the new array compared to the old one
print(k2_len, k2_f_len)
print(k3_len, k3_f_len)

print("{} short tweets were found and removed for the keyword 'gift'".format(k1_len - k1_f_len))
print("{} short tweets were found and removed for the keyword 'science'".format(k2_len - k2_f_len))
print("{} short tweets were found and removed for the keyword 'actor'".format(k3_len - k3_f_len))

print('-----------------------------------------------------------------')
print('The first 5 tweets for \"{}\":'.format(keywords[0]))
print('-----------------------------------------------------------------\n')
[print("{}: {}\n".format(i + 1, k1_tweets_filtered[i]['text'])) for i in range(5)]

print('\n-----------------------------------------------------------------')
print('The first 5 tweets for \"{}\":'.format(keywords[1]))
print('-----------------------------------------------------------------\n')
[print("{}: {}\n".format(i + 1, k2_tweets_filtered[i]['text'])) for i in range(5)]

print('\n-----------------------------------------------------------------')
print('The first 5 tweets for \"{}\":'.format(keywords[2]))
print('-----------------------------------------------------------------\n')
[print("{}: {}\n".format(i + 1, k3_tweets_filtered[i]['text'])) for i in range(5)] 

def remove_non_ascii(s): return "".join(i for i in s if ord(i)<128)
def pre_process(doc):
    """
    pre-processes a doc
      * Converts the tweet into lower case,
      * removes the URLs,
      * removes the punctuations
      * tokenizes the tweet
      * removes words less that 3 characters
    """
    
    doc = doc.lower()
    # getting rid of non ascii codes
    doc = remove_non_ascii(doc)
    
    # replacing URLs
    url_pattern = "http://[^\s]+|https://[^\s]+|www.[^\s]+|[^\s]+\.com|bit.ly/[^\s]+"
    doc = re.sub(url_pattern, 'url', doc) 

    punctuation = r"\(|\)|#|\'|\"|-|:|\\|\/|!|\?|_|,|=|;|>|<|\.|\@"
    doc = re.sub(punctuation, ' ', doc)
    
    return [w for w in doc.split() if len(w) > 2]

tweet_k1 = k1_tweets_filtered[0]['text'] #calls our previous function
tweet_k1_processed = pre_process(tweet_k1)

print(tweet_k1)
# tweet_k1_processed is now a list of words. 
# We use ' '.join() method to join the list to a string.
print(' '.join(tweet_k1_processed))

tweet_k2 = k2_tweets_filtered[0]['text']
tweet_k2_processed = pre_process(tweet_k2)

print("First tweet of k2_tweets_filtered: {}\n".format(tweet_k2))
print("First tweet of k2_tweets_filtered (processed): {}\n".format(' '.join(tweet_k2_processed)))

tweet_k3 = k3_tweets_filtered[0]['text']
tweet_k3_processed = pre_process(tweet_k3)

print("First tweet of k3_tweets_filtered: {}\n".format(tweet_k3))
print("First tweet of k3_tweets_filtered (processed): {}\n".format(' '.join(tweet_k3_processed)))

k1_tweets_processed = [pre_process(tweet['text']) for tweet in k1_tweets_filtered]
k2_tweets_processed = [pre_process(tweet['text']) for tweet in k2_tweets_filtered]
k3_tweets_processed = [pre_process(tweet['text']) for tweet in k3_tweets_filtered]

print('-----------------------------------------------------------------')
print('The first 5 processed tweets for k1_tweets_processed:')
print('-----------------------------------------------------------------\n')
[print("{}: {}\n".format(i + 1, ' '.join(k1_tweets_processed[i]))) for i in range(5)]

print('-----------------------------------------------------------------')
print('The first 5 processed tweets for k2_tweets_processed:')
print('-----------------------------------------------------------------\n')
[print("{}: {}\n".format(i + 1, ' '.join(k2_tweets_processed[i]))) for i in range(5)]

print('-----------------------------------------------------------------')
print('The first 5 processed tweets for k3_tweets_processed:')
print('-----------------------------------------------------------------\n')
[print("{}: {}\n".format(i + 1, ' '.join(k3_tweets_processed[i]))) for i in range(5)]

def construct_termdoc(docs, vocab=[]):
   
   
    
    # vocab is not passed
    if vocab == []:
        vocab = set()
        termdoc_sparse = []

        for doc in docs:       
            # computes the frequencies of doc
            doc_sparse = Counter(doc)    
            termdoc_sparse.append(doc_sparse)
            
            # update the vocab
            vocab.update(doc_sparse.keys())  

        vocab = list(vocab)
        vocab.sort()
    
    else:
        termdoc_sparse = []        
        for doc in docs:
            termdoc_sparse.append(Counter(doc))
            

    n_docs = len(docs)
    n_vocab = len(vocab)
    termdoc_dense = np.zeros((n_docs, n_vocab), dtype=int)

    for j, doc_sparse in enumerate(termdoc_sparse):
        for term, freq in doc_sparse.items():
            try:
                termdoc_dense[j, vocab.index(term)] = freq
            except:
                pass
            
    return termdoc_dense, vocab

k1_termdoc, k1_vocab = construct_termdoc(k1_tweets_processed)

# print out the term-by-document matrix
print(k1_termdoc)
# print out the first 5 vocabularies
print(' '.join(k1_vocab[:5]))  # print out only the first 5 vocabs

# visualise the term-by-document matrix
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(k1_termdoc)
ax.set_xlabel('term (vocabulary)')
ax.set_ylabel('documents (tweets)')
ax.set_title('Term-by-Document matrix from tweets collected for keyword \"{}\"'.format(keywords[0]))

def Euclidean_distance(x,y):
    '''
    Compute and return the Euclidean distance between two vectors x and y
    '''
    try:
        e_d = np.sqrt(np.sum((x - y) ** 2))
        return e_d
    except Exception as e:
        print(e)
        return None

def compute_euclidean_distance_matrix(termdoc):
    
    euclidean_distance_matrix = np.empty([termdoc.shape[0], termdoc.shape[0]])
    
    for i in range(termdoc.shape[0]):
        for j in range(termdoc.shape[0]):
            euclidean_distance_matrix[i, j] = Euclidean_distance(termdoc[i, :], termdoc[j, :])
    
    return euclidean_distance_matrix

# compute the distance matrix for k1_termdoc using the function "compute_euclidean_distance_matrix"
k1_euclidean_distances = compute_euclidean_distance_matrix(k1_termdoc)

# Visualise the distance matrix for this keyword
fig, ax = plt.subplots(figsize=(7, 7))
k1i = ax.imshow(k1_euclidean_distances)
k1cb = fig.colorbar(k1i)