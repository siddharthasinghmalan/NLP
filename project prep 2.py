#STOP WORDS


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is an example showing of stop word filteration  "
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

filtered_sentence = [ w for w in words if not w in stop_words]

##This is the explanation of above line of code 
##
## filteres_sentence =[]
##
##for w in words :
##  if w not in stop_words:
##      filtered_sentence.append(w)
## 

print(filtered_sentence)