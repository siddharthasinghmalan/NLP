import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC , LinearSVC , NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifeir(ClassifierI):
    def __init__(self , *classifiers):
        self.classifeirs = classifiers

    def classify(self ,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self , features ):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        

        choice_votes = votes.count(mode(votes))
        conf = choice_votes /len(votes)




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
#top 89300 words we are going to check against
word_features = list(all_words.keys())[:89300]

def find_features(document):
    words = set(document)   
    features = {}
    for w in word_features:
        #features is a dictionary
        # if the word is in top 3000 words it is true else false
        features[w] = w in words

    return features

#print((find_features ("short_reviews/negative.txt")))

#rev = review 
featuresets = [(find_features(rev) , category) for (rev, category) in documents]
# use the 1st 89300 words and use the trained data against the file (negative.txt) whether the words have +ve impact or -ve         

#negative data
training_set = featuresets[:4000]
testing_set = featuresets[1000:]


#classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier_f = open("naivebayes.pickle" , "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()




print("Original Naive AByes Algo accuracy percent :", (nltk.classify.accuracy(classifier , testing_set)) * 100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent :", (nltk.classify.accuracy(MNB_classifier , testing_set)) * 100)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB classifier accuracy percent :", (nltk.classify.accuracy(BernoulliNB_classifier , testing_set)) * 100)

####
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent :", (nltk.classify.accuracy(LogisticRegression_classifier , testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent :", (nltk.classify.accuracy(SGDClassifier_classifier , testing_set)) * 100)


SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent :", (nltk.classify.accuracy(SVC_classifier , testing_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("Linear_classifier accuracy percent :", (nltk.classify.accuracy(LinearSVC_classifier , testing_set)) * 100)


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent :", (nltk.classify.accuracy(NuSVC_classifier , testing_set)) * 100)


voted_classifier = VoteClassifeir(
    classifier , 
    NuSVC_classifier,
    LinearSVC_classifier , 
    SVC_classifier , 
    SGDClassifier_classifier ,
    LogisticRegression_classifier ,
    BernoulliNB_classifier , 
    MNB_classifier
    )

print("voted_classifier accuracy percent :", (nltk.classify.accuracy(voted_classifier , testing_set)) * 100)


