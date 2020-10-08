# What constitutes a chick-flick? Analyzing gender preference in movie plot keywords using natural language processing in python
This project analyzes gender, keywords, revenue and ratings for movies from multiple sources. 

538 data related to gender-specific ratings for IMDB reviews can be found here: https://fivethirtyeight.com/features/what-if-online-movie-ratings-werent-based-almost-entirely-on-what-men-think/


IMDB revenue and key-word plot synopsis data sourced from here: https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata

# Table of Contents
1. [Data Preparation](#data-preparation)
2. [Random Forest Model](#random-forest-model)
3. [Neural Network](#neural-network)
4. [Keyword Visualizations](#keyword-visualizations)
5. [Additional Analyses](#additional-analyses)
6. [Conclusion](#conclusion)

# Data Preparation
#### import libraries
```python
import json
import kaggle
import pandas as pd 
import re
import numpy as np
import spacy
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
from nltk.stem.porter import PorterStemmer
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.datasets import load_hobbies
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import plotly.express as px
from collections import defaultdict
```
### Data cleaning
##### load and combine data
```python
gender = pd.read_csv("gender_ratings.csv")
gender=gender.replace("’", "'", regex=True)
join_table = pd.read_csv("movies_metadata.csv")
movie_ids=join_table[['id', 'title', 'revenue', 'release_date']]
keywords = pd.read_csv("keywords.csv")
```

##### reformat keyword data
```python
##convert list of dictionaries in columns to an array of keywords
column_of_lists=[]
for i in range(len(keywords)): 
    row=keywords.loc[i, "keywords"]
    try:
        dictionaries = json.loads(row.replace("'", "\""))
        keyword_list=[]
        for Dict in dictionaries:
            keyword=(Dict['name'])
            keyword_list.append(keyword)
    except:
        keyword_list.append([])
    column_of_lists.append(keyword_list)
    
keywords['word_array']=column_of_lists
keywords['id']=keywords['id'].astype(str)
```

##### some leftover string cleaning to make two datasets compatible 
```python
MovieDict = {
"Goodfellas":"GoodFellas"
,"Your Name":"Your Name."
,"Barfi!":"Barfi"
,"Fistful of Dollars":"A Fistful of Dollars"
,"Harry Potter and the Sorcerer's Stone":"Harry Potter and the Philosopher's Stone"
,"Terminator 2":"Terminator 2: Judgment Day"
,"Catch Me if You Can":"Catch Me If You Can"
,"Gone With the Wind":"Gone with the Wind"
,"Jagten":"Jagten"
,"Amores Perros":"Amores perros"
,"X: First Class":"X-Men: First Class"
,"The Girl With the Dragon Tattoo":"The Girl with the Dragon Tattoo"
,"Birdman or (The Unexpected Virtue of Ignorance)":"Birdman"
,"Dances With Wolves":"Dances with Wolves"
,"Yip Man":"Yip Man chinchyun"
,"Star Trek: Into Darkness":"Star Trek Into Darkness"
,"Rogue One":"Rogue One: A Star Wars Story"
,"Contratiempo":"Contratiempo"
,"The Boy in the Striped Pajamas":"The Boy in the Striped Pyjamas"
,"Chak de! India":"Chak De! India"
,"The Man From Earth":"The Man from Earth"
,"Letters From Iwo Jima":"Letters from Iwo Jima"
,"Mandariinid":"Mandariinid"}

def year_title_recode(row):
    if row == 'V for Vendetta':
        return '2005'
    elif row == 'Kingsman: The Secret Service':
        return '2016'
    elif row == 'The Fall':
        return '2006'
    elif row == 'Ex Machina':
        return '2014'
```

##### merge dataframes
```python
gender['title']=gender['title'].map(MovieDict).fillna(gender['title'])
df = pd.merge(keywords, movie_ids, on='id', how='left')

df['release_date'] =  pd.to_datetime(df['release_date'], format='%Y-%m-%d')
df['year'] = df['release_date'].dt.year.astype('object')
gender['year'] = gender['year'].astype('object')

df['year'] = df['title'].apply(year_title_recode).fillna(df['year'])
gender['year'] = gender['title'].apply(year_title_recode).fillna(gender['year'])

data = pd.merge(df, gender, on=['title','year'], how='outer')

### clean strings
data['word_array']=data['word_array'].astype(str)
data=data.replace("\\[", "", regex=True)
data=data.replace("\\]", "", regex=True)
data=data.replace("\\'", "", regex=True)
data=data.replace(">", "", regex=True)
data['Actual']=data['Actual'].astype(float)
data.dropna(subset=['Actual'], how='all', inplace=True)
data.reset_index(inplace=True)
```
### descriptive statistics
##### gender_class variable is coded such that '1' refers to movies that women ranked higher than men and '0' refers to movies that men ranked higher than women
```python
data['gender_class'] = np.where((data['womenonly'] > data['menonly']), 1, 0) 
data['gender_class_factor'] = np.where(data['gender_class']==1, 'women', 'men') 
data[["gender_class_factor"]].describe()

data[["gender_class_factor", "revenue"]].groupby("gender_class_factor").mean()
```

gender        | count   |Avg. revenue  |
------------- | --------|--------------|
men           | 207     | $339,787,900 |
women         | 294     | $111,100,700 |



##### prep for supervised learning model
```python

nltk.download('stopwords')

word_array_cleaned = []
###words only no phrases
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data.loc[i,'word_array'])
    review = review.lower()
    review = review.split(" ")
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    word_array_cleaned.append(review)

data['word_array_cleaned']=word_array_cleaned

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features = 5000)
X=cv.fit_transform(data['word_array_cleaned']).toarray()
y=data['gender_class'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

```
# Random Forest Model
##### model creation
```python
# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators =100,criterion="entropy",random_state =13)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)[:,1]

```
##### random forest model evaluation
```python
######### model evaluation ###########
# Fitting classifier to the Training set
# Compute and print the confusion matrix and classification report
cm=pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted men', 'Predicted women'],
    index=['men', 'women']
)
print(cm)
print(classification_report(y_test, y_pred))

acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

cv_auc = cross_val_score(classifier, X_test, y_test, cv=10, scoring='roc_auc')
print(cv_auc)

## write predictions to dataframe for later analysis
data['predictions']=classifier.predict_proba(X) [:,1]
```
![alt text](https://github.com/cfrench575/movie-ratings-nlp/blob/master/images/randomforest_CM.png)


# Neural Network
##### exploring parameter options
```python
target = to_categorical(y_train)

### parameter tuning - activation functions and learning rates
activation_functions=['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
optimizers=['sgd', 'adam', 'rmsprop']
accuracies=[]
acts=[]
optims=[]

for actfunc in activation_functions: 
    for opt in optimizers: 
        model = Sequential()
        model.add(Dense(12, input_dim=2600, activation=actfunc))
        model.add(Dense(8, activation=actfunc))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        hist=model.fit(X_train, target, epochs=25, batch_size=10, validation_split = 0.2)
        accuracy=max(hist.history['val_accuracy'])
        accuracies.append(accuracy)
        acts.append(actfunc)
        optims.append(opt)
        nods.append(node)

print(model_stats=list(zip(acts, optims, accuracies)))

### parameter tuning: nodes and epochs

nodes=[10, 20, 50, 100, 200]
batches=[10, 20, 30, 40, 50]
accuracies=[]
ns=[]
bs=[]

for node in nodes: 
    for batch in batches:
        model = Sequential()
        model.add(Dense(node, input_dim=2600, activation='exponential'))
        model.add(Dense(node, activation='exponential'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        hist=model.fit(X_train, target, epochs=25, batch_size=batch, validation_split = 0.2)
        accuracy=max(hist.history['val_accuracy'])
        accuracies.append(accuracy)
        ns.append(node)
        bs.append(batch)

print(model_stats=list(zip(ns, bs, accuracies)))

```
##### final model
```python
model = Sequential()
model.add(Dense(20, input_dim=2600, activation='exponential'))
model.add(Dense(10, activation='exponential'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, target, epochs=10, batch_size=10, validation_split = 0.2)
```
##### final model evaluation: additional validation set
```python
y_pred = model.predict_classes(X_test)

cm=pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Predicted men', 'Predicted women'],
    index=['men', 'women']
)
print(cm)
print(classification_report(y_test, y_pred))

acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 
```
![alt text](https://github.com/cfrench575/movie-ratings-nlp/blob/master/images/neuralnet_validation.png)

# Keyword Visualizations
### visualizations of common keywords for men and women
##### clean keywords as phrases instead of single words for better interpretability 
```python
word_array_disp = []
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z,]', '', data.loc[i,'word_array'])
    review = review.lower()
    review = review.split(",")
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    word_array_disp.append(review)
    
data['word_array_disp']=word_array_disp
````
##### create keyword frequency tables for women-preferred versus men-preferred movies based on gender separated rankings
````python
men_df = data[data["gender_class"] ==0]
women_df = data[data["gender_class"] ==1]

word_freq_men = defaultdict(int)
for text in men_df['word_array_disp']:
    for word in text.split():
        word_freq_men[word] += 1

word_freq_women = defaultdict(int)
for text in women_df['word_array_disp']:
    for word in text.split():
        word_freq_women[word] += 1

word_freq = defaultdict(int)
for text in data['word_array_disp']:
    for word in text.split():
        word_freq[word] += 1

word_men=pd.DataFrame.from_dict(word_freq_men, orient='index') \
    .sort_values(0, ascending=False) \
    .rename(columns={0: 'abs_freq_men'})

word_women=pd.DataFrame.from_dict(word_freq_women, orient='index') \
    .sort_values(0, ascending=False) \
    .rename(columns={0: 'abs_freq_women'})

word_total=pd.DataFrame.from_dict(word_freq, orient='index') \
    .sort_values(0, ascending=False) \
    .rename(columns={0: 'abs_freq_total'})
    
word_freq_array=pd.concat([word_men, word_women, word_total], axis=1)
word_freq_array["perc_men"] = (word_freq_array["abs_freq_men"]/word_freq_array["abs_freq_total"]) *100
word_freq_array["perc_women"] = (word_freq_array["abs_freq_women"]/word_freq_array["abs_freq_total"])*100
word_freq_array["rel_diff"] = (word_freq_array["perc_men"]-word_freq_array["perc_women"])
````
##### frequency of keywords for movies that men rated more highly than women. The frequency of the same keywords for movies that women rated more highly than men are included for comparison
```python
table_df_men=word_freq_array[['abs_freq_men', 'abs_freq_women']].dropna(subset=['abs_freq_men']).sort_values(by=['abs_freq_men']).tail(10)

import matplotlib.pyplot as plt
table_men = table_df_men.plot.barh(rot=0, stacked=True)
plt.show()

```
![alt text](https://github.com/cfrench575/movie-ratings-nlp/blob/master/images/men_freq.png)

##### frequency of keywords for movies that women rated more highly than men. The frequency of the same keywords for movies that men rated more highly than women are included for comparison
```python

table_df_women=word_freq_array[['abs_freq_women', 'abs_freq_men']].dropna(subset=['abs_freq_women']).sort_values(by=['abs_freq_women']).tail(10)

table_women = table_df_women.plot.barh(rot=0, stacked=True)
plt.show()
```
![alt text](https://github.com/cfrench575/movie-ratings-nlp/blob/master/images/women_freq.png)

### visualizations of most gender-differentiating keywords
##### best gender-differentiating keywords for men - keywords with the greatest difference in frequency between movies that men rated more highly than women
```python
table_df_mendiff=word_freq_array[['perc_men', 'perc_women', 'rel_diff' ]].dropna(subset=['rel_diff']).sort_values(by=['rel_diff']).tail(10)

table_mendiff = table_df_mendiff[['perc_men', 'perc_women' ]].plot.barh(rot=0, stacked=True)
plt.show()
```
![alt text](https://github.com/cfrench575/movie-ratings-nlp/blob/master/images/men_difference.png)

##### best gender-differentiating keywords for women - keywords with the greatest difference in frequency between movies that women rated more highly than men
```python
table_df_womendiff=word_freq_array[['perc_men', 'perc_women', 'rel_diff' ]].dropna(subset=['rel_diff']).sort_values(by=['rel_diff'], ascending= False).tail(10)

table_womendiff = table_df_womendiff[['perc_women','perc_men' ]].plot.barh(rot=0, stacked=True)
plt.show()
```
![alt text](https://github.com/cfrench575/movie-ratings-nlp/blob/master/images/women_difference.png)

# Additional Analyses
#### correlation between rating and revenue for men vs women 
```python
dfr = data[["gender_class_factor", 'menonly', 'womenonly', 'revenue']]
corr = dfr.corr()
print(corr)
````
![alt text](https://github.com/cfrench575/movie-ratings-nlp/blob/master/images/corr_matrix.png)

#### weighted frequency of keywords by revenue
```python
def word_frequency(text_list, num_list, sep=None):
    word_freq = defaultdict(lambda: [0, 0])
    for text, num in zip(text_list, num_list):
        for word in text.split(sep=sep): 
            word_freq[word][0] += 1 
            word_freq[word][1] += num
    columns = {0: 'abs_freq', 1: 'wtd_freq'}
    abs_wtd_df = (pd.DataFrame.from_dict(word_freq, orient='index')
                 .rename(columns=columns )
                 .sort_values('wtd_freq', ascending=False)
                 .assign(rel_value=lambda df: df['wtd_freq'] / df['abs_freq']).round())
    abs_wtd_df.insert(1, 'abs_perc', value=abs_wtd_df['abs_freq'] / abs_wtd_df['abs_freq'].sum())
    abs_wtd_df.insert(2, 'abs_perc_cum', abs_wtd_df['abs_perc'].cumsum())
    abs_wtd_df.insert(4, 'wtd_freq_perc', abs_wtd_df['wtd_freq'] / abs_wtd_df['wtd_freq'].sum())
    abs_wtd_df.insert(5, 'wtd_freq_perc_cum', abs_wtd_df['wtd_freq_perc'].cumsum())
    return abs_wtd_df
    
weighted_rev_frequency= weighted_rev_frequency[['wtd_freq']].sort_values(by=['wtd_freq']).tail(10)

table_weighted_rev_frequency = weighted_rev_frequency.plot.barh(rot=0)
plt.show()
````
![alt text](https://github.com/cfrench575/movie-ratings-nlp/blob/master/images/revenue_weighted_freq.png)

# Conclusion

* out of 500 movies with ratings separated by gender, women ranked the same movie higher than the men in the sample 294 times. In other words there were 294 movies that were ranked higher for women only compared to men only

* Movies that women ranked more favorably than men made less revenue than movies that men ranked more favorably than women (339.787 million on average for men compared to 1.111 million on average for women)

* Movie rank was not correlated with revenue for men (r= .09), but revenue was negatively correlated with rank for women (r=-.39). For women, a more favorable (lower number) ranking for a movie was associated with more revenue.


corr matrix| menonly   |  womenonly |  revenue |
-----------|-----------|------------|----------|
menonly    | 1.000000  |  0.438230  |-0.094386 |
womenonly  | 0.438230  |  1.000000  |-0.394175 |
revenue    | -0.094386 |  -0.394175 | 1.000000 |


* Both a random forest model and a neural net were able to use plot keywords to classify whether or not a movies was ranked more favorably by men compared to women with some degree of accuracy. Cross-validation scores for each model yielded a wide range of accuracies (60%-80%) though the models performed better than chance. Both models were able to classify which movies would be ranked for favorably for women than for men (around 90%), but were less accurate when classifying which movies would be ranked more favorably by men compared to women (around 40%).

* Most frequent keywords associated with **men-preferred** movies are: 

keyword            | frequency |  
------------------ | ----------|
basedonnovel       | 25        |
duringcreditssting | 24        |
aftercreditssting  | 21        |
magic              | 16        |
anim               | 14        |

* Most gender-differentiated keywords associated with **men-preferred** movies are: 

keyword                  | % men-preferred movies |  
-------------------------| -----------------------|
aftercreditssting        | 95.5%                  |
marvelcom                | 91.7%                  |
magic                    | 88.9%                  |
wifehusbandrelationship  | 85.7%                  |
neighbor                 | 85.7%                  |


* Most frequent keywords associated with **women-preferred** movies are: 

keyword        | frequency |  
---------------|-----------|
murder         | 21        |
dystopia       | 20        |
dyinganddeath  | 17        |
violence       | 14        |
basedonnovel   | 13        |

* Most gender-differentiated keywords associated with **women-preferred** movies are: 

keyword         | % women-preferred movies |  
----------------| -------------------------|
filmnoir        | 91.7%                    |
assassin        | 90.9%                    |
crime           | 87.5%                    |
detect          | 87.5%                    |
theft           | 85.7%                    |
