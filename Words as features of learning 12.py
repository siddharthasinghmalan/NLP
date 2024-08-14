import nltk
import random
from nltk.corpus import movie_reviews


#for training , words from movie reviews
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)

#print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())



all_words = nltk.FreqDist(all_words)
#top 3000 words we are going to check against
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)   
    features = {}
    for w in word_features:
        #features is a dictionary
        # if the word is in top 3000 words it is true else false
        features[w] = w in words

    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

#rev = review 
featuresets = [(find_features(rev) , category) for (rev, category) in documents]
# use the 1st 3000 words and use the trained data against the file (29416.txt) whether the words have +ve impact or -ve         