import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB

from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC, LinearSVC , NuSVC   

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self , *classifiers):
        self._classifiers = classifiers

    def classify(self , features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self , features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf




#for training , words from movie reviews
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]


#random.shuffle(documents)

#print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())



all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)   
    features = {}
    for w in word_features:
        features[w] = w in words

    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev) , category) for (rev, category) in documents]
               

#positive data example
training_set = featuresets[:1900]
testing_set = featuresets[1900:]


#negative data set
training_set = featuresets[100:]
testing_set = featuresets[:100]

#Bayes algorithm
#posterior = prior occurences * likelihood / evidence

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original NaiveBayes ALgo accuracy :" , (nltk.classify.accuracy(classifier , testing_set))*100)
classifier.show_most_informative_features(15)

#save_classifier = open("naivebayes.pickle","wb")
#pickle.dump(classifier , save_classifier)
#save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy :" , (nltk.classify.accuracy(classifier , testing_set))*100)

#GaussianNB , BernoulliNB
#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print("Gaussian_classifier accuracy :" , (nltk.classify.accuracy(GaussianNB_classifier , testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy :" , (nltk.classify.accuracy(BernoulliNB_classifier , testing_set))*100)

#LogisticRegression , SGDClassifier
#SVC, LinearSVC , NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy :" , (nltk.classify.accuracy(LogisticRegression_classifier , testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy :" , (nltk.classify.accuracy(SGDClassifier_classifier , testing_set))*100)

#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print("SVC_classifier accuracy :" , (nltk.classify.accuracy(SVC_classifier , testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy :" , (nltk.classify.accuracy(LinearSVC_classifier , testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy :" , (nltk.classify.accuracy(NuSVC_classifier , testing_set))*100)






###################









voted_classifier = VoteClassifier(classifier , 
                                  MNB_classifier , 
                                  BernoulliNB_classifier , 
                                  LogisticRegression_classifier , 
                                  SGDClassifier_classifier , 
                                  LinearSVC_classifier , 
                                  NuSVC_classifier )
print("Voted classifier accuracy :" , (nltk.classify.accuracy(voted_classifier , testing_set))*100)

#print("Classification :" ,voted_classifier.classify(testing_set[0][0]), "Confidence % :" , voted_classifier.confidence(testing_set[0][0]))


#print("Classification :" ,voted_classifier.classify(testing_set[1][0]), "Confidence % :" , voted_classifier.confidence(testing_set[1][0]))
#print("Classification :" ,voted_classifier.classify(testing_set[2][0]), "Confidence % :" , voted_classifier.confidence(testing_set[2][0]))
#print("Classification :" ,voted_classifier.classify(testing_set[3][0]), "Confidence % :" , voted_classifier.confidence(testing_set[3][0]))
#print("Classification :" ,voted_classifier.classify(testing_set[4][0]), "Confidence % :" , voted_classifier.confidence(testing_set[4][0]))
#print("Classification :" ,voted_classifier.classify(testing_set[5][0]), "Confidence % :" , voted_classifier.confidence(testing_set[5][0]))