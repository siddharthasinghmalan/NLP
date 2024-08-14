from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

#Default parameter for lemmatize is :       pos="n"

print(lemmatizer.lemmatize("cats"))   
print(lemmatizer.lemmatize("cacti"))  
print(lemmatizer.lemmatize("rocks"))  
print(lemmatizer.lemmatize("geese"))  
print(lemmatizer.lemmatize("python"))  
print(lemmatizer.lemmatize("better",pos="a"))  
print(lemmatizer.lemmatize("best",pos="a"))
