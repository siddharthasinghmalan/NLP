import nltk
import random
from nltk.corpus import movie_reviews


#for training , words from movie reviews
#list of tuples
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)

#print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())


#top15 most common words
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))

#no. of times stupid word has been used
print(all_words["stupid"])