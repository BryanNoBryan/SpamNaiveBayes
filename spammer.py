#ML and plotting
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#One hot encoding
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
#import pandas
import pandas as pd
#to split the array
import numpy as np

#Auxillary
import math
from collections import OrderedDict

#preprocessing
import re
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize

data = pd.read_csv("spam.csv")
data.drop(data.columns[[0, 1]], axis=1, inplace=True)

tokenized_list = []

# Removing punctuations in string
# Using regex
for i in range(len(data)):
    #remove punctuation
    data.iloc[i, 0] = re.sub(r'[^\w\s]', '', data.iloc[i, 0])
    #remove escape chars and numbers
    data.iloc[i, 0] = re.sub(r'[\n\r]|[\d]', ' ', data.iloc[i, 0])
    # remove URLs
    data.iloc[i, 0] = re.sub(r'https?://\S+|www\.\S+', ' ', data.iloc[i, 0])


    # lemmatized_list = [lemmatizer.lemmatize(token) for token in data.iloc[i, 0].split(' ')]
    tokenized_string = word_tokenize(data.iloc[i, 0])
    lemmatized_list = [lemmatizer.lemmatize(token).lower() for token in tokenized_string]

    #use string not generator
    tokenized_list.append(' '.join(lemmatized_list))

# print(tokenized_list)

vectorizor = CountVectorizer()
cv_matrix = vectorizor.fit_transform(tokenized_list)

vectored_data = pd.DataFrame(data=cv_matrix.toarray(),columns = vectorizor.get_feature_names_out())

X_train, X_test, y_train, y_test = train_test_split(vectored_data, data.iloc[:,1], test_size = 0.20, random_state = 0)

#NAIVE BAYES BEGIN
#https://www.analyticsvidhya.com/blog/2021/01/a-guide-to-the-naive-bayes-algorithm/ 

#getting probability for if SPAM

val_count = y_train.value_counts()
spam_length = val_count[1]
ham_length = val_count[0]
prior_prob_spam = val_count[1] / (val_count[0] + val_count[1])
print('prior_prob_spam: ' + str(float(prior_prob_spam)))

spam_word_freq = {}
spamicity = {}
for col in X_train.columns:
    spam_word_freq[col] = 0

length = len(X_train)

for col in X_train.columns:
    for index, value in X_train[col].items():
        # is spam
        if (y_train[index] == 1 & value > 0):
            #col is the word
            spam_word_freq[col] += 1
    #smoothing
    spamicity[col] = (spam_word_freq[col]+1)/(length+2)

    print(f"Number of spam emails with the word {col}: {spam_word_freq[col]}")
    print(f"Spamicity of the word '{col}': {spamicity[col]} \n")

print('calced spamicity')

ham_word_freq = {}
hamicity = {}
for col in X_train.columns:
    ham_word_freq[col] = 0

for col in X_train.columns:
    for index, value in X_train[col].items():
        # is ham
        if (y_train[index] == 0 & value > 0):
            #col is the word
            ham_word_freq[col] += 1
    #smoothing
    hamicity[col] = (ham_word_freq[col]+1)/(length+2)

print('calced hamicity')

naive_bayes = {}
for col in X_train.columns:
    PWordSpam = spam_word_freq[col]
    PSpam = prior_prob_spam
    PWordHam = ham_word_freq[col]
    PHam = 1 - PSpam
    # https://medium.com/@insight_imi/sms-spam-classification-using-na%C3%AFve-bayes-classifier-780368549279
    naive_bayes[col] = (PWordSpam * PSpam) / ((PWordSpam * PSpam) + (PWordHam * PSpam))

print('calced naive bayes')
print(naive_bayes)

for i in range(len(X_test)):
    row = pd.DataFrame(X_test.iloc[i, :]).transpose()
    org_index = row.index[0]
    prob = 1
    for name, data in row.items():
        if (data > 0):
            try:
                val = naive_bayes[name]
                prob *= (val != 0) if val else 1/(spam_length+2)
            except KeyError:
                prob *= 1/(spam_length+2)
    print(f"case {i}: {prob}    {y_test[org_index]}")
    


# try:
#             pr_WS = spamicity[word]
#         except KeyError:
#             pr_WS = 1/(total_spam+2)  # Apply smoothing for word not seen in spam training data, but seen in ham training 
#             print(f"prob '{word}' is a spam word: {pr_WS}")








# print(data.head())

# print(data.describe(include = 'all'))

# print(data.groupby('label_num').describe())