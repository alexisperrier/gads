# L13 - NLP Notebook 1


import pandas as pd
from bs4 import BeautifulSoup
import re



train = pd.read_csv("../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print(train["review"][0])
# remove html tags
example1 = BeautifulSoup(train["review"][0], "html.parser")

print(train["review"][0])
print(example1.get_text())

# remove ponctuation and numbers
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print(letters_only)

# Tokenize
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words

import nltk
nltk.download()  # Download text data sets, including stop words

from nltk.corpus import stopwords # Import the stop word list
print(stopwords.words("english"))

# remove stop words
words = [w for w in words if not w in stopwords.words("english")]


def review_to_words( raw_review ):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))

    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # 6. Join the words back into one string separated by space,  and return the result.
    return( " ".join( meaningful_words ))

num_reviews = train["review"].size
clean_train_reviews = []

for i in range( 0, 1000 ):
    if( (i+1)%100 == 0 ):
        print("Review %d of %d\n" % ( i+1, 1000 ))
    clean_train_reviews.append( review_to_words( train["review"][i] ) )


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
print(vocab)

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)