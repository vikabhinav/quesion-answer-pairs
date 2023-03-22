from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os
import string
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df1 = pd.read_csv('/content/drive/MyDrive/GL/Capstone/Data/S08_question_answer_pairs.txt', sep='\t')
df2 = pd.read_csv('/content/drive/MyDrive/GL/Capstone/Data/S09_question_answer_pairs.txt', sep='\t')
df3 = pd.read_csv('/content/drive/MyDrive/GL/Capstone/Data/S10_question_answer_pairs.txt', sep='\t', encoding = 'ISO-8859-1')

df2.head() # display records in df2

df3.head() # display records in df3

all_data=df1.append([df2,df3]) # Merge data in both df2, df3 to "all_data"

# display all details of "all_data" [ with details like column number, number of records and each column/attribute names] 
all_data.info()

# Modify the data in column "Question" 
# --> First in column "ArticleTitle" replace underscore with space 
# --> second merge one more space with the data from "Question" in the end
all_data['Question'] = all_data['ArticleTitle'].str.replace('_', ' ') + ' ' + all_data['Question']

# eliminate all Columns except 'Question', 'Answer'
all_data = all_data[['Question', 'Answer']]
all_data.shape #display size (number of row / columns)

all_data.head(10) #display top 10 records

# remove duplicate question / or repeating question
all_data = all_data.drop_duplicates(subset='Question')
all_data.head(10) #display top 10 records


all_data.shape #display size (number of row / columns)


all_data = all_data.dropna() # drop null records
all_data.shape #display size (number of row / columns)


import nltk
nltk.download('stopwords')

# create stopword to remove stop words such as “the”, “a”, “an”, “in”
stopwords_list = stopwords.words('english') 

# lemmatizer to remove repeated words with same meaning
lemmatizer = WordNetLemmatizer()

# define function to split the sentences to usuable words
def my_tokenizer(doc):
    words = word_tokenize(doc) # to split a sentance to words (Tokenizes a sentence into words and punctuation)
    
    pos_tags = pos_tag(words) # for providing specific tags for certain words
    
    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list] # remove stop words / filter out non-stop words
    
    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation] # remove punctuations
    
    lemmas = []
    
    # as per the words in POS Tag, differentiate each words to Adjuctive, verb, Noun...
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos = wordnet.ADJ 
        elif w[1].startswith('V'):
            pos = wordnet.VERB
        elif w[1].startswith('N'):
            pos = wordnet.NOUN
        elif w[1].startswith('R'):
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN
        
        lemmas.append(lemmatizer.lemmatize(w[0], pos)) # remove lemmatize or similar words with same meaning

    return lemmas
    
    
    
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# Convert the obtained collection of raw documents to a matrix
tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)
tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(all_data['Question']))
print(tfidf_matrix.shape)


# define function to input the 
def ask_question(question):
    query_vect = tfidf_vectorizer.transform([question]) # convert the question sentence to vector matrix
    similarity = cosine_similarity(query_vect, tfidf_matrix) # get Cosine simarity of the question matrix vs the dataset matrix(which is already created)
    max_similarity = np.argmax(similarity, axis=None) # get the word with maximum similarity
    
    # print the question asked
    print('Your question:', question) 
    # print question from dataset which is closer to question asked using the word with Max similarity.
    print('Closest question found:', all_data.iloc[max_similarity]['Question']) 
    # print the accuracy of similarity between the questions
    print('Similarity: {:.2%}'.format(similarity[0, max_similarity]))
    # print the corresponding answer from dataset
    print('Answer:', all_data.iloc[max_similarity]['Answer'])
    


ask_question('When Abraham Lincoln started his political career')

ask_question('Where was Nicola Tesla born')

ask_question('Can whales fly')

ask_question('Who was the third president of the United States')

ask_question('How high are crime rates in Brazil')