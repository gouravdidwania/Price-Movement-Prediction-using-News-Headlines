<!-- PROJECT LOGO -->
![maxresdefault](https://user-images.githubusercontent.com/86877457/132740281-f1784d38-cd9a-4a9b-8fbe-e3b3099b4008.jpg)
<br />
<p align="center">

  <h3 align="center">Price Movement Prediction using News Headlines</h3>

  <p align="center">
    In this project, I'll try predicting price movement direction of Dow Jones Industrial Average (DJIA) using python, which can be applied on various other stocks and financial instruments like Futures, Options and Bonds .
    <br />
  </p>
</p>




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#problem-statement">Problem Statement</a>
    </li>
    <li>
	<a href="#data-overview">Data Overview</a>
	<ul>
          <li><a href="#data-attributes">Data Attributes</a></li>
          <li><a href="#data-snapshot">Data Snapshot</a></li>
        </ul>
    </li>
    <li><a href="#implementaion">Implementaion</a>
	<ul>
          <li><a href="#data-preparation">Data Preparation</a></li>
          <li><a href="#classification-modelling">Classification Modelling</a></li>
	  <li><a href="#best-baseline-models">Best Baseline Models</a></li>
	  <li><a href="#performance-on-test-data">Performance on Test Data</a></li>
	  <li><a href="#auc-roc-analysis">AUC ROC Analysis</a></li>
        </ul>
    </li>
    <li><a href="#final-thoughts">Final Thoughts</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

We are living in a constantly changing world. Every day the “breaking news” could impact our decision making on our lives. It is especially the case for stock market. I am curious if the news headlines of the day could affect the stock market close value of the same day.

**Dow Jones Industrial Average** is a stock market index that collects the value of a list of 30 large and public companies based in the US. In this way it gives an idea of the trend that is going through the stock market.

News and global events, political or otherwise, play a major role in changing stock values. Every stock exchange is, after all, reflects how much trust investors are ready to put in other companies.

<!-- PROBLEM STATEMENT -->
## Problem Statement

The overall aim of this process is to identify dependency of price of the stock on daily news headlines and whether the headlines could predict whether the stock closing price will increase or decrease.

Insights from this can be used to develop systems for traders to decide whether one should hold positions in the market or not.


<!-- DATA OVERVIEW -->
## Data Overview

Source: [Kaggle](https://www.kaggle.com/aaron7sun/stocknews)

Data: This data contains 8 years of daily news headlines from 2000 to 2016 from Yahoo Finance and Reddit WorldNews, as well as the Dow Jones Industrial Average(DJIA) close value of the same dates as the news. This will be a binary classification problem. When the target is “0”, the same day DJIA close value decreased compared with the previous day, when the target is “1”, the value rose or stayed the same.

Downloaded the csv file from Kaggle.

### Data Attributes

**1. Date:** Date column contains the dates from 2000 to 2016 on which the news are released.

**2. Label:** Binary Numeric, '0' represent that the price went down and '1' represent that the price went up.

**3. Top#:** Strings that contains the top 25 news headlines for the day ranging from 'Top1' to 'Top25'

### Data Snapshot

Let’s look at the data. We have a date column, a label column, and 25 top news columns. Each row represents a day.

![image](https://user-images.githubusercontent.com/86877457/132743780-556bd0a8-2b50-477b-9129-306d66f1c5ae.png)

<!-- IMPLEMANTATION -->
## Implementaion

**Real-world/Business objectives and constraints**

- No low-latency requirement.
- Errors can be costly.
- Should be robust in nature.

### Data Preparation

**Data Preparation Steps:**

1. Remove date column
2. Combine 25 top news headlines of each day into one document
3. Clean the text data by converting abbreviations to full form, removing punctuations & stop words, lowering the letter, and performing lemmatizations.
4. Tokenize each document into list of words
5. Use 3 word embedding methods to convert the tokens into numeric values: Bag of words, word TF-IDF and GoogleNews word2vec then doc2vec.

**Data Cleaning**

We need to remove punctuations and stop words, also lower the letters. Before doing these, the abbreviations need to be converted to regular words because their meaning could be lost while cleaning. I made this function to filter out all the abbreviations from the entire dataset.

  ```sh
  def find_abbr(text):
    abbr = []
    for i in re.finditer(r"([A-Za-z]+| )([A-Za-z]\.){2,}", text):
        abbr.append(i.group())
    df_abbr = pd.Series(abbr)
    return df_abbr.unique()	
  ```
Here is the result:

![image](https://user-images.githubusercontent.com/86877457/132745464-238c5005-c2aa-4b81-8871-8ee3b76898b1.png)

The abbreviations which were important were converted to their full forms other abbreviation like A.K.A were removed. Then a column was created 'headlines_str' which contained all the headlines merged into one string. 

I created a loop that can convert raw text into a list of word tokens. This function tokenise the string into word tokens. Further cleaning is performed to lemmatise the words, remove stop words as well as lower the letters. Here is the loop.

  ```sh
  df['headlines_str'] = np.nan
  for i in range(len(df.Headlines)):
    df.headlines_str[i] = df.Headlines[i].lower()
    df.headlines_str[i] = re.sub("[^a-zA-z]"," ",df.headlines_str[i])
    words = nltk.word_tokenize(df.headlines_str[i])
    words = [wl.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    df.headlines_str[i] = ' '.join(words)	
  ```
After this, the headlines were converted from left to right:

![image](https://user-images.githubusercontent.com/86877457/132746901-78788bdb-aac1-480a-a276-c6a57f3d79f9.png)

The words were stored in the form of a lists in seperate column 'headlines_words' as shown below:

![image](https://user-images.githubusercontent.com/86877457/132746710-83dee565-0ed6-42b5-806e-27f377c63610.png)

**Data Preparation**

Machine learning models don’t understand human words, but they know vectors!

I want to try several different word embedding methods, because they could affect the performance of the models. For bag of words, TF-IDF embedding, we need to have each document as a long string. For word2vec embedding, the vectors will be what we need.

- Doc2vec: The famous GoogleNews word2vec model was trained on 3 billion running words and contains 3 million 300-dimension English word vectors. I downloaded the model from [here](https://github.com/mmihaltz/word2vec-GoogleNews-vectors). Our prediction will be based on each document instead of words. So made a document vector by averaging the word vectors on each combined string in 'headlines_str'. I created a function to do this task. After running each document through this function, every document is now a 300 dimensional vector.

  ```sh
  def doc2vec(model, wordlist):
    '''
    Use a word2vec embedding model to get the vecter of each word of the wordlist.
    Now we have a list of vecters, len(list)= number of words in the doc, len(vector)= the model type, e.g.300
    Convert each doc into a vector by np.mean. len(doc vec) = 300
    '''
    # Filter the list of vectors to include only those that Word2Vec has a vector for
    vector_list = [model[word] for word in wordlist if word in model.vocab]
    doc_vector = np.mean(vector_list, axis=0)
    return doc_vector
  ```
Now we need to create a new data frame that has all the relevant data format we need for modelling later which we can convert it into a pickle file. Here is the review.

![image](https://user-images.githubusercontent.com/86877457/132747924-e6fd6987-1c14-42c5-a1c1-b0467d8f24c8.png)

After the above preprocessing, we have the data ready for modelling.

### Classification Modelling

Every dataset is different. We don’t know which model works best for our data until we try all of them. Here I selected 7 different classification models. These models are all classic that usually works well in filtering baseline models. I know from my previous projects that Naive Bayes and Random Forest works good in such situations. Let's find out which works best.

Each embedding method will have 7 reports from the 7 models. After comparing the results, we have the best baseline model:

**Train - Validation - Test Split**

This is a time series data where we have to predict future results with data of past so randomly dividing the data shouldn't be performed. So, I put away data since 2015 beginning as test data, and further split the rest of data from 2000 to 2014 as test data and 2014 to 2015 as validation data. In order to make the process concise and clean, I created a function to try all the models.

  ```sh
  from sklearn import  svm, naive_bayes, neighbors, ensemble
  from sklearn.linear_model import LogisticRegression
  
  lr_model = LogisticRegression()
  nb_model = naive_bayes.GaussianNB()
  knn_model = neighbors.KNeighborsClassifier()
  svc_model = svm.SVC(probability=True, gamma="scale")
  rf_model = ensemble.RandomForestClassifier(n_estimators=100)
  et_model = ensemble.ExtraTreesClassifier(n_estimators=100)
  ada_model = ensemble.AdaBoostClassifier()
  
  models = ["lr_model", "nb_model", "knn_model", "svc_model", "rf_model", "et_model", "ada_model"]
  ```
  ```sh
  def baseline_model_filter(modellist, X, y):
    ''' 1. split the train data further into train and validation (17%). 
        2. fit the train data into each model of the model list
        3. get the classification report based on the model performance on validation data
    '''
    X_train, X_valid, y_train, y_valid = X[:3471],X[3471:],y[:3471],y[3471:]
    for model_name in modellist:
        curr_model = eval(model_name)
        curr_model.fit(X_train, y_train) 
        print(f'{model_name} \n report:{classification_report(y_valid, curr_model.predict(X_valid))}')
```
  
We are going to try the 7 models on data prepared in four different word embedding methods. Bag of words, word TF-IDF and word2vec. Each method will give me 7 reports, I selected the best performing model and listed them below

### Best Baseline Models

**For Bad of Words/Count Vectorizer:**
---------------------------------------

```sh
count_vect = CountVectorizer(analyzer='word')
X = count_vect.fit_transform(df_train.news_str).toarray()
y = df_train.Label
baseline_model_filter(models, X, y)
```
Naive Bayes and Random Forest model gives the best result for bag of words embedding.

![image](https://user-images.githubusercontent.com/86877457/132750464-283b21df-1f9a-47f9-88c2-5168bdee840d.png)

![image](https://user-images.githubusercontent.com/86877457/132750553-ce22ceb2-32fe-4555-9372-4a6ec096f7da.png)

**For Word TF-IDF:**
----------------------

KNN model gives the best result for TF-IDF embedding. However, the bag of words result is slightly better than word level TF-IDF overall.

![image](https://user-images.githubusercontent.com/86877457/132750729-7a07b26d-4fe1-42fc-83fd-dd4c0ee34257.png)

This three models were selected considering the general model performance and f1 scores. Word2Vec were not that promising

### Performance on Test Data

The model mentioned above were selected and **hyper-parameter tuning** was performed for each model.

**GridSearchCV** was used to find the best hyper paramters.

__**Naive Bayes Model with Count Vectorizer**__

The Confusion Matrix and the Classification Report for the following is shown below:

![image](https://user-images.githubusercontent.com/86877457/132752524-556bdf65-1d73-4aba-ac2f-664950c70366.png)

__**Random Forest with Count Vectorizer**__

The Confusion Matrix and the Classification Report for the following is shown below:

![image](https://user-images.githubusercontent.com/86877457/132752723-b9673415-9c02-47e2-b0c1-8a8d39bb01fb.png)

__**KNN Model with TFID Vectorizer**__

The Confusion Matrix and the Classification Report for the following is shown below:

![image](https://user-images.githubusercontent.com/86877457/132752813-eeb10dbf-8610-4b4c-94cf-1a944382a78e.png)

The Results were surprisingly different. Best Model obtained during the training process resulted the least perfromance. We'll be selecting **Naive Bayes Model** in comparision to 'Random Forest' as both accuracy and the F1 score for the classification are better.

### AUC ROC Analysis

An Investor/Trader invest a lot of money in the market and saving money is more important than earning money for them. The purpose of this Analysis was to reduce the **False Positive Rate**, so that the Investor/Trader can save money by pulling out positions in the model's false claims.

1. At first, the probabilty is predicted corresponding to each test data.
2. True Positive Rate, False Positive Rate and False Negative Rate for range of threshold from 0 to 1 is calculated.
3. These values are plotted on the graph and the perfect threshold is selected. This graph obtained is as shown:

![image](https://user-images.githubusercontent.com/86877457/132754479-7ae62a40-c2a6-4ea5-b7cc-c945e3dfcff1.png)

4. Threshold ~ 0.7 seems good from the graph and final after testing 0.72 was selected. 

![image](https://user-images.githubusercontent.com/86877457/132754716-d5772bd7-9f4b-45a9-8e02-b7a6b3116635.png)

**Training accuracy ~ 72%**

**Test accuracy ~ 85%**

This was the final result obtained.

<!-- FINAL THOUGHTS -->
## Final Thoughts

I was curious whether daily news deadline could affect the stock market price of the same day. So I started this project. After working on raw text data processing and classification modelling, I got the best performing model, Naive Bayes with Bag of Words embedding. This model successfully predicted 85 % of the situation when the stock price increased or stayed the same. Stock price prediction is a very complex problem. There are so many affecting elements. At least now we know that news headlines are definitely one them!

**FURTHER IMPROVEMENT**
- The current analysis is based on 16 years of data from 2000 to 2016. I would like to collect more data and recent data to improve the model.
- News from single relavent source will prove more efficient in building the model. It will ease cleaning and produce more quality data for modelling.
- The classification models I used in this project are classic machine learning models. In the future I would like to try deep learning model because deep learning works very well in natural language processing problems.
