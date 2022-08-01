---
layout: post
title:  "Sentiment Classification with the Naive Bayes Algorithm"
date:   2022-05-02 -
description: "Natural language processing (NLP) is an important branch of machine learning that is concerned with developing programs that enable computers the ability to process and analyze natual language data. Applications of NLP tasks can be observed in speech recognition, text-to-speech, recommender systems, document classification and summerization along with various other examples. Given the vast sources of available text data, it is important to develop the skills to process and analyze text data. This post explores the Naive Bayes algorithm applied on a document classification task."
categories: Python Natural_Languange_Processing Document_Classification Naive_Bayes Bag_of_Words
html_url: /assets/img/NaiveBayes/Conditional Probabilities.webp
---

**Outline**
-   [Introduction](#introduction)
-   [Naive Bayes](#naive-bayes)
-   [Document Classification Example](#example-classification-example)
-   [IMDB Dataset Application](#imdb-dataset-application)

## Introduction

Natural language processing (NLP) is undoubtably one of the most important branches of machine learning. NLP-based technologies are becoming increasingly widespread across various business sectors; thus humans are interfacing with these technologies at an increasing rate. Virtual assistants, chatbots, text summarization, document classification, speech recognition, language translators are all examples of NLP applications that are frequently used. Given the prominence of NLP-based applications, it is important to develop an understanding of NLP concepts to be able to develop programs that allow for text data to be analyzed and utilized in various applications. 

This post will demonstrate basic text processing steps commonly encountered in NLP tasks. These steps will be applied on a synthetic dataset to map text into numerical data which will then be used to fit a naive Bayes classification model and make predictions on unobserved data. The framework of the naive Bayes model will also be briefly explained and then will be applied on the IMBD movie review dataset to generate a sentiment classifier. This model will be integrated using a web application generated using Dash for users to classify the sentiment of reviews. 

## The Naive Bayes Algorithm

The naive Bayes algorithm derives its name from the key assumption made by the model. The model assumes that the model features are conditionally independent given the class label. Within the scope of NLP, the naive Bayes assumption does not seem suitable given that words in a structured sentence are dependent on previous sequences of words. Despite this apparent issue, the naive Bayes algorithm has been implemented successfully in document classification applications like email spam filtering. 

The conditional independence assumption greatly simplifies how the class conditional distribution is specified. Namely, the class conditional density can be expressed as the product of one dimensional densities as shown below.

$$
\begin{align*}

\small p(\mathbf{x}|y=c, \mathbf{\theta}) = \prod_{j=1}^{D}p(x_{j}|y=c, \mathbf{\theta}_{jc})

\end{align*}
$$

Without conditional independence, the expression above would instead be written as

$$
\begin{align*}

\small p(\mathbf{x}|y=c, \mathbf{\theta}) = \prod_{j=1}^{D}p(x_{j}|\bigcap x_{1:j-1}, y=c, \mathbf{\theta}_{jc})

\end{align*}
$$

This expression is the joint distribution defined by the **chain rule** of probability. This minor change greatly increases the computational and memory demands that would be required by the model. Under this representation of the joint density, a D-dimensional table of conditional probabilities would be required. With datasets containing many features, this requirement may be computationally prohibitive.

The exact form of the class conditional density will depend on the feature. In this example, the features will be a vector of counts representing each document. This means that the class conditional density will be of the form of a multinomial distribution thus

$$
\begin{align*}

\small p(\mathbf{x}|y=c, \mathbf{\theta}) = \text{Mu}(\mathbf{x}_{i}|N_{i}, \mathbf{\theta}_{c}) = \frac{N_{i}!}{\prod_{j=1}^{D}x_{ij}!}\prod_{j=1}^{D}\theta_{jc}^{x_{ij}}

\end{align*}
$$

where $$\small N_i$$ is the number of terms in each document and $$\small \theta_c$$ is a vector of probabilities which indicate the probability of generating word $$\small j$$ in documents of class $$\small c$$. These parameters will be shown in greater detail in the following section. Before moving forward, a brief digression must first be taken to explain an important computational detail that arises when implementing a generative classifier like the naive Bayes algorithm.

It is often the case that generative classifiers will deal with high-dimensional vectors. Observing the equation above, consider a scenario where the vector of probabilities is a $$\small 5000$$-dimensional vector. The equation above specifies that each entry in the probabilities vector be raised to a power that is defined by the number of times the word $$\small j$$ appears in a particular document and then be multiplied by each individual probability in the vector. The numerous multiplications of small probabilities will lead to numerical underflow causing the implementation of the naive Bayes algorithm to fail. To fix this, the probabilities must be mapped into the log domain. Although this remedies the numerical underflow issue, another issue will arise when the probability of observing a particular vector is required to be determined. 

To calculate the probability, Bayes' rule is applied as follows

$$
\begin{align*}

\small p(y=c|\mathbf{x}) = \frac{p(\mathbf{x}|y=c)p(y=c)}{\sum_{c'}p(\mathbf{x}|y=c')p(y=c')}

\end{align*}
$$

mapping into the log domain yields

$$
\begin{align*}

\small \log p(y=c|\mathbf{x}) = \log p(\mathbf{x}|y=c) + \log p(y=c) - \log \sum_{c'}p(\mathbf{x}|y=c')p(y=c')

\end{align*}
$$

notice that this requires that the log transform of the denominator be evaluated; however, because the log is outside of the sum, it prohibits the denominator from being transformed into the log domain. To work around this issue, the **log-sum-exp** trick will be used. To demonstrate this trick, the denominator is expressed as follows

$$
\begin{align*}

\small \log \sum_{c'}p(\mathbf{x}|y=c')p(y=c') & \; \small = \log \sum_{c'}\exp(\log p(\mathbf{x}|y=c') + \log p(y=c')) \\[1.5ex]

\end{align*}
$$

This allows for the largest term in the denominator to be factored out and then the remaining numbers can be expressed relative to that. The following example demonstrates this statement.

$$
\begin{align*}

\small \log(\exp(-120) + \exp(-121)) & \; \small = \log(\exp(-120)(\exp(0) + \exp(-1))) \\[1.5ex] 

& \; \small = \log(\exp(0) + \exp(-1)) - 120

\end{align*}
$$

This can be expressed generally as

$$
\begin{align*}

\small \log \sum_{c} \exp{b_c} = \log(\sum_{c} \exp{b_{c} - B}) + B

\end{align*}
$$

This will be demonstrated in the following section.

## Document Classification Example

Document classification is an NLP task in which the goal is to classify a document (web page, email, news article, etc.) into a class from a set of classes. This is done on a dataset where the inputs are the text contained in a document with an associated label. Thus the goal is to learn a mapping from the inputs $$\small \mathbf{x}$$ to outputs $$\small y$$ where $$\small y \in \{1,\cdots, C\}$$. Consider the following inputs and outputs.

```python
example_data = [
    "Investors sell shares admist uncertainty causing the S&P 500 to drop.",
    "The broad stock market index fell 1.6%, with increasing losses.",
    "Economic data shows an increase of inflation. \
    Supply-chain disruptions have caused inflation to rise, putting banks in \
    tough positions.",
    "The S&P 500 has dropped siginificantly and is the worst four months in decades.",
    "Benzema delivers a special moment after scoring a panenka at the Ethiad. Despite \
    defensive issues, Real Madrid manages to stay alive in the competition after playing \
    out a 4-3 thriller that will surely be an instant classic in Champions League football.",
    "It was a match which will be remembered as a Champions League classic. There were seven goals \
    one outrageous penalty, plenty of chances and outstanding performances as Manchester \
    City and Real Madrid played out a thrilling first-leg semifinal tie."
]

example_labels = [
    "finance",
    "finance",
    "finance",
    "finance",
    "sports",
    "sports"
]
```

A total of six documents constitutes the training set and there are two classes, namely *finance* and *sports*. The first task is to process the text data and build a corpus. Text is usually processed to remove punctuation, stop words, numbers, and to normalize the text. To understand why text is preprocessed, consider the words *run*, *Run*, *run!*, and *running*. Although in certain sentences these words convey the same meaning, they are all different in python. This is seen if the following comparisons are made.

```python
"run" == "Run"
False

"run" == "run!"
False
```

It is trivial that *run* and *running* are not the same but consider a scenario in which it may make sense to reduce running to its root word. This will likely depend on the syntactic function of the word; thus one will need to know the parts-of-speech class that a particular word belongs to. Luckily, there are existing tools in python that will make this an easy task. The following demonstration is intended to cover the basic steps in text preprocessing. 

First, import the necessary libraries

```python
import string
import re
import nltk
```

The first processing task will be to remove all punctuation.

```python
def punctuation_removal(data):
    punctuation = set(string.punctuation)
    processed_text = [" " if word in punctuation else word for datum in data for word in datum]
    processed_text = " ".join("".join(processed_text).split())
    return processed_text

punctuation_removal(["##This is an$$ example!!!"])

'This is an example'
```

Next, a list representing the corpus is made.

```python
def build_corpus(text):
    corpus = text.split()
    return corpus

build_corpus('This is an example')

['This', 'is', 'an', 'example']
```

The following steps remove all numbers, turn uppercase words into lower case, remove stop words, and remove all words that contain less than three characters.

```python
def words_only(corpus):
    corpus = "".join([re.sub("[^a-zA-Z]", " ", text) for text in corpus]).split()
    corpus = [text.strip() for text in corpus]
    return corpus

def lower_case(corpus):
    corpus = [text.lower() for text in corpus]
    return corpus

def stopwds_removal(corpus):
    stopwds = set(nltk.corpus.stopwords.words("english"))
    corpus = [text for text in corpus if text not in stopwds]
    return corpus

def length_check(corpus, l=3):
    corpus = [text for text in corpus if len(text) >= 3]
    return corpus
```

Lastly, the corpus is lemmatized using nltk's parts-of-speech tagging tool in conjunction with their lemmatizer tool. This function is a bit more complex, but its main functions are to tag the corpus, convert the tags into valid arguments for the lemmatizer and then transform the corpus according to its respective parts-of-speech.

```python
def lemmatize_corpus(corpus):
    tagged_corpus = nltk.pos_tag(corpus)
    
    VALID_TAGS = ["n", "v", "a", "r", "s"]
    adjective_tags = set(["JJ", "JJR", "JJS"])
    adverb_tags = set(["RB", "RBR", "RBS"])
    noun_tags = set(["NN", "NNP", "NNPS", "NNS"])
    verb_tags = set(["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])
        
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    
    lemmatized_corpus = []
    for text, tag in tagged_corpus:
        if tag in adjective_tags:
            tag = "a"
            lemmatized_corpus.append(lemmatizer.lemmatize(text, pos=tag))
        elif tag in adverb_tags:
            tag = "r"
            lemmatized_corpus.append(lemmatizer.lemmatize(text, pos=tag))
        elif tag in noun_tags:
            tag = "n"
            lemmatized_corpus.append(lemmatizer.lemmatize(text, pos=tag))
        elif tag in verb_tags:
            tag = "v"
            lemmatized_corpus.append(lemmatizer.lemmatize(text, pos=tag))
            
    return lemmatized_corpus
```

Lastly, the following two functions combine the preprocessing functions to create a corpus for the model and to process the data without losing the identity of each respective document. 

```python
def process_text(data):
    text = punctuation_removal(data)
    corpus = build_corpus(text)
    corpus = words_only(text)
    corpus = lower_case(corpus)
    corpus = stopwds_removal(corpus)
    corpus = length_check(corpus)
    corpus = lemmatize_corpus(corpus)
    return corpus

def process_data(data):
    processed_data = []
    for datum in data:
        processed_data.append(" ".join(process_text(datum)))
    return processed_data
```

Applying the preprocessing onto the example data, the following is obtained.

```python
#build corpus and process text data
example_corpus = process_text(example_data)
example_processed_data = process_data(example_data)

example_processed_data

['investor sell share admist uncertainty cause drop',
 'broad stock market index fell increase loss',
 'economic data show increase inflation supply chain disruption cause \
 inflation rise put bank tough position',
 'drop siginificantly bad month decade',
 'benzema delivers special moment score panenka defensive issue real madrid \
 manages stay alive competition play thriller surely instant classic champion \
 league football',
 'match remember champion league classic goal outrageous penalty plenty chance \
 outstanding performance manchester city real madrid play thrill first leg \
 semifinal tie']
```

compare this to the original data

```python
example_data = [
    "Investors sell shares admist uncertainty causing the S&P 500 to drop.",
    "The broad stock market index fell 1.6%, with increasing losses.",
    "Economic data shows an increase of inflation. \
    Supply-chain disruptions have caused inflation to rise, putting banks in \
    tough positions.",
    "The S&P 500 has dropped siginificantly and is the worst four months in decades.",
    "Benzema delivers a special moment after scoring a panenka at the Ethiad. Despite \
    defensive issues, Real Madrid manages to stay alive in the competition after playing \
    out a 4-3 thriller that will surely be an instant classic in Champions League football.",
    "It was a match which will be remembered as a Champions League classic. There were seven goals \
    one outrageous penalty, plenty of chances and outstanding performances as Manchester \
    City and Real Madrid played out a thrilling first-leg semifinal tie."
]
```

Using the corpus, a matrix of word counts is constructed. This matrix will represent how many times a word that is contained in the corpus appears in the document. Thus, each word in the document will be compared to the corpus and a running count will be tracked until every word in the document has been examined. The following function accomplishes this task.

```python
def build_design_matrix(corpus, data):
    matrix = {}
    documents = len(data)
    for text in sorted(corpus):
        matrix[text] = [0] * documents
    for i in range(documents):
        for word in data[i].split():
            if word in matrix:
                matrix[word][i] += 1
                
    return matrix


example_design_matrix_dict = build_design_matrix(example_corpus, example_processed_data)
example_design_matrix_dict

{'admist': [1, 0, 0, 0, 0, 0],
 'alive': [0, 0, 0, 0, 1, 0],
 'bad': [0, 0, 0, 1, 0, 0],
 'bank': [0, 0, 1, 0, 0, 0],
 'benzema': [0, 0, 0, 0, 1, 0],
 'broad': [0, 1, 0, 0, 0, 0],
 'cause': [1, 0, 1, 0, 0, 0],
 'chain': [0, 0, 1, 0, 0, 0],
 'champion': [0, 0, 0, 0, 1, 1],
 'chance': [0, 0, 0, 0, 0, 1],
 'city': [0, 0, 0, 0, 0, 1],
 'classic': [0, 0, 0, 0, 1, 1],
 'competition': [0, 0, 0, 0, 1, 0],
 'data': [0, 0, 1, 0, 0, 0],
 'decade': [0, 0, 0, 1, 0, 0],
 'defensive': [0, 0, 0, 0, 1, 0],
 'delivers': [0, 0, 0, 0, 1, 0],
 'disruption': [0, 0, 1, 0, 0, 0],
 'drop': [1, 0, 0, 1, 0, 0],
 'economic': [0, 0, 1, 0, 0, 0],
 'fell': [0, 1, 0, 0, 0, 0],
 'first': [0, 0, 0, 0, 0, 1],
 'football': [0, 0, 0, 0, 1, 0],
 'goal': [0, 0, 0, 0, 0, 1],
 'increase': [0, 1, 1, 0, 0, 0],
 'index': [0, 1, 0, 0, 0, 0],
 'inflation': [0, 0, 2, 0, 0, 0],
 'instant': [0, 0, 0, 0, 1, 0],
 'investor': [1, 0, 0, 0, 0, 0],
 'issue': [0, 0, 0, 0, 1, 0],
 'league': [0, 0, 0, 0, 1, 1],
 'leg': [0, 0, 0, 0, 0, 1],
 'loss': [0, 1, 0, 0, 0, 0],
 'madrid': [0, 0, 0, 0, 1, 1],
 'manages': [0, 0, 0, 0, 1, 0],
 'manchester': [0, 0, 0, 0, 0, 1],
 'market': [0, 1, 0, 0, 0, 0],
 'match': [0, 0, 0, 0, 0, 1],
 'moment': [0, 0, 0, 0, 1, 0],
 'month': [0, 0, 0, 1, 0, 0],
 'outrageous': [0, 0, 0, 0, 0, 1],
 'outstanding': [0, 0, 0, 0, 0, 1],
 'panenka': [0, 0, 0, 0, 1, 0],
 'penalty': [0, 0, 0, 0, 0, 1],
 'performance': [0, 0, 0, 0, 0, 1],
 'play': [0, 0, 0, 0, 1, 1],
 'plenty': [0, 0, 0, 0, 0, 1],
 'position': [0, 0, 1, 0, 0, 0],
 'put': [0, 0, 1, 0, 0, 0],
 'real': [0, 0, 0, 0, 1, 1],
 'remember': [0, 0, 0, 0, 0, 1],
 'rise': [0, 0, 1, 0, 0, 0],
 'score': [0, 0, 0, 0, 1, 0],
 'sell': [1, 0, 0, 0, 0, 0],
 'semifinal': [0, 0, 0, 0, 0, 1],
 'share': [1, 0, 0, 0, 0, 0],
 'show': [0, 0, 1, 0, 0, 0],
 'siginificantly': [0, 0, 0, 1, 0, 0],
 'special': [0, 0, 0, 0, 1, 0],
 'stay': [0, 0, 0, 0, 1, 0],
 'stock': [0, 1, 0, 0, 0, 0],
 'supply': [0, 0, 1, 0, 0, 0],
 'surely': [0, 0, 0, 0, 1, 0],
 'thrill': [0, 0, 0, 0, 0, 1],
 'thriller': [0, 0, 0, 0, 1, 0],
 'tie': [0, 0, 0, 0, 0, 1],
 'tough': [0, 0, 1, 0, 0, 0],
 'uncertainty': [1, 0, 0, 0, 0, 0]}
```

The model parameters will constitute of the prior $$\small \hat{\pi_c}$$ and the conditional class probabilities $$\small \hat{\mathbf{\theta}_c}$$. These are typically estimated as

$$
\begin{align*}

\small \hat{\pi_c} = \frac{N_c}{N}

\end{align*}
$$

and 

$$
\begin{align*}

\small \hat{\theta_{jc}} = \frac{N_{jc}}{N_c}

\end{align*}
$$

Unfortunately, using this method to estimate conditional probabilities will likely lead to the naive Bayes algorithm to fail. This is because there exists the possibility for the $$\small j$$-word to not appear in a document. This means that the conditional probability will be zero and thus taking the log of zero or multiplying by zero (if one is not working in the log domain) will result in the algorithm to fail. Instead, Laplace smoothing will be used to estimate the conditional probabilities. Thus 

$$
\begin{align*}

\small \hat{\theta_{jc}} = \frac{N_{jc} + \alpha}{N_c + \alpha * D}

\end{align*}
$$

The following function calculates the priors and conditional probabilities.

```python
def fit(X, y, alpha=1):
    
    #N = number of documents in the training data
    #D = number of features
    N, D = X.shape
    y = np.array(y)
    
    #C = classes
    #N_c = counts of each class
    C, N_c = np.unique(y, return_counts=True)
    log_priors = np.log(N_c / N)
    
    log_Theta_c = {}
    for c in C:
        x_i = np.sum(X[y == c], axis=0) + alpha
        N_ic = np.sum(np.sum(X[y == c], axis=0)) + (alpha * D)
        log_Theta_c[c] = np.log(x_i / N_ic)
        
    return log_priors, log_Theta_c
```

This function takes the design matrix and the labels as arguments, but the design matrix must be a numpy array. This is done as follows

```python
import numpy as np
example_design_matrix = np.array(list(example_design_matrix_dict.values())).T

example_design_matrix 

array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0],
       [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,
        0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        1, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0],
       [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
        0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1,
        1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0]])
```

The output array is the numeric representation of the original text data. Next, the model is fit

```python
example_priors, example_log_Theta_c = fit(example_design_matrix, example_labels)

example_priors, example_log_Theta_c

(array([-0.40546511, -1.09861229]),
 {'finance': array([-3.93182563, -4.62497281, -3.93182563, -3.93182563, -4.62497281,
         -3.93182563, -3.52636052, -3.93182563, -4.62497281, -4.62497281,
         -4.62497281, -4.62497281, -4.62497281, -3.93182563, -3.93182563,
         -4.62497281, -4.62497281, -3.93182563, -3.52636052, -3.93182563,
         -3.93182563, -4.62497281, -4.62497281, -4.62497281, -3.52636052,
         -3.93182563, -3.52636052, -4.62497281, -3.93182563, -4.62497281,
         -4.62497281, -4.62497281, -3.93182563, -4.62497281, -4.62497281,
         -4.62497281, -3.93182563, -4.62497281, -4.62497281, -3.93182563,
         -4.62497281, -4.62497281, -4.62497281, -4.62497281, -4.62497281,
         -4.62497281, -4.62497281, -3.93182563, -3.93182563, -4.62497281,
         -4.62497281, -3.93182563, -4.62497281, -3.93182563, -4.62497281,
         -3.93182563, -3.93182563, -3.93182563, -4.62497281, -4.62497281,
         -3.93182563, -3.93182563, -4.62497281, -4.62497281, -4.62497281,
         -4.62497281, -3.93182563, -3.93182563]),
  'sports': array([-4.71849887, -4.02535169, -4.71849887, -4.71849887, -4.02535169,
         -4.71849887, -4.71849887, -4.71849887, -3.61988658, -4.02535169,
         -4.02535169, -3.61988658, -4.02535169, -4.71849887, -4.71849887,
         -4.02535169, -4.02535169, -4.71849887, -4.71849887, -4.71849887,
         -4.71849887, -4.02535169, -4.02535169, -4.02535169, -4.71849887,
         -4.71849887, -4.71849887, -4.02535169, -4.71849887, -4.02535169,
         -3.61988658, -4.02535169, -4.71849887, -3.61988658, -4.02535169,
         -4.02535169, -4.71849887, -4.02535169, -4.02535169, -4.71849887,
         -4.02535169, -4.02535169, -4.02535169, -4.02535169, -4.02535169,
         -3.61988658, -4.02535169, -4.71849887, -4.71849887, -3.61988658,
         -4.02535169, -4.71849887, -4.02535169, -4.71849887, -4.02535169,
         -4.71849887, -4.71849887, -4.71849887, -4.02535169, -4.02535169,
         -4.71849887, -4.71849887, -4.02535169, -4.02535169, -4.02535169,
         -4.02535169, -4.71849887, -4.71849887])})
```

With the priors and conditional class probabilities, predictions can now be made. The following function uses the log priors and conditional class log probabilities to make predictions

```python
def predict(X, log_priors, log_Theta_c):
    C = list(log_Theta_c.keys())
    predictions = []
    for i in range(X.shape[0]):
        log_probability = np.array([])
        for j, c in enumerate(C):
            log_likelihood = np.sum(log_Theta_c[c] * X[i])
            log_posterior = log_likelihood + log_priors[j]
            log_probability = np.append(log_probability, log_posterior)
        predictions.append(C[np.argmax(log_probability)])
    return predictions
```

This function will be applied to the following unseen data.

```python
example_test_data = [
    "Kevin De Bruyne's superb diving header gave City the lead after just 94 seconds. \
    When Gabriel Jesus spinned inside the box to double the home side's lead in the \
    11th minute it seemed as if Real was in for an arduous evening. The 13-time champion, \
    the most successful club in the competition's history, had never before conceded two \
    goals so quickly in the Champions League.",
    "The S&P 500 has dropped 13% so far this year, and many experts are turning more \
    bearish toward stocks."
]
```

It is obvious that the first example belongs to the sports category and the second belongs to finance. Before this data is passed into the predict function, it must be processed as the training data was to obtain a correct numeric representation. 

```python
#process test data
example_processed_test_data = process_data(example_test_data)
example_design_matrix_dict = build_design_matrix(example_corpus, example_processed_test_data)
example_design_matrix = np.array(list(example_design_matrix_dict.values())).T
```

Now the `example_design_matrix` is passed into the predict function along with the class priors and class conditional probabilities. 

```python
predict(example_design_matrix, example_log_priors, example_log_Theta_c)

['sports', 'finance']
```

The correct predictions were made. Note that the `predict` function makes the prediction by selecting the maximum unnormalized log probabilities. This makes sense when the probability of a certain class overwhelms the remaining probabilities but what if the class probabilities were $$\small 0.5001$$ and $$\small 0.4999$$? Blindly selecting the maximum probability without knowledge of its magnitude may lead to issues if a certain model were implemented in a business application. For example, consider an adult content blocker designed to prevent children from accessing websites with adult-related content. Now, if a particular website has been classified as acceptable with a probability of $$\small 50.01\%$$. One would not be willing to risk their child accessing this website despite the website being classifies as acceptable. 

To obtain the related probabilities, the previously mentioned **log-sum-exp** trick will be utilized. The predict function is modified by adding the **log-sum-exp** trick as follows

```python
def predict_log_probabilities(X, log_priors, log_Theta_c):
    C = list(log_Theta_c.keys())
    log_probabilities = []
    #calculate unnormalized log probabilities
    for i in range(X.shape[0]):
        log_probability = np.array([])
        for j, c in enumerate(C):
            log_likelihood = np.sum(log_Theta_c[c] * X[i])
            log_posterior = log_likelihood + log_priors[j]
            log_probability = np.append(log_probability, log_posterior)
        #log-sum-exp_trick
        B = max(log_probability)
        marginal_likelihood = np.log(np.sum(np.exp(log_probability - B))) + B
        log_probability = log_probability - marginal_likelihood
        log_probabilities.append(log_probability)
    return np.array(log_probabilities)

    log_probs = predict_log_probabilities(example_design_matrix, example_log_priors, example_log_Theta_c)

    np.exp(log_probs)

    array([[0.00590484, 0.99409516],
       [0.93535158, 0.06464842]])
```

The output indicates that the first document is classified as a `sports` article with a probability of $$\small 99.41 \% $$ and the second document is classified as a `finance` article with a probability of $$\small 93.53 \% $$. This outlines the process of document classification. Text from the training data is first processed and a corpus is created. A matrix of word counts is then generated and then the class priors and conditional probabilities are calculated which are then used to classify new documents. This process will now be applied on a larger dataset to generate a sentiment-classification model which will then be integrated with a simple web application that will classify unseen reviews as positive or negative. 

## IMDB Movie Review Dataset Classification

The IMDB movie review dataset contains $$\small 25,000$$ reviews that are labeled by sentiment (positive/negative). The sentiment classification will allow for the model to associate words that belong to each respective class. Once a model has been trained, the model will be able to classify unseen text. The data must be extracted from the text files and assembled into a dataset. Once this is completed the preprocessing functions are applied. 

```python
corpus = process_text(data_train)
processed_data = process_data(data_train)
design_matrix_dict = build_design_matrix(corpus, processed_data)
len(design_matrix_dict.keys())

64667
```

Notice that the corpus consists of $$\small 64,667$$ distinct words. To make this data matrix slightly more manageable, the words in the final matrix will be limited. The subset will be selected randomly for simplicity; however, in practice, one may consider selecting a subset that is comprised with words that commonly occur. This will yield a corpus that is more informative and thus may yield a more accurate model. The following code block limits the corpus to $$\small 10,000$$ words.

```python
key_num = len(design_matrix_dict.keys())
samples = 10000
keys = np.array(list(design_matrix_dict.keys()))
index = np.random.choice(key_num, samples, replace=False)
corpus_keys = sorted(np.take(keys, index))
final_design_matrix_dict = {}
for key in corpus_keys:
    final_design_matrix_dict[key] = design_matrix_dict[key]
design_matrix = np.array(list(final_design_matrix_dict.values())).T
```

With the final array obtained, the model is fit and the model class log priors and conditional densities are computed. Typically, one would evaluate the accuracy of the model against a validation set to tune model parameters and once an acceptable model accuracy is achieved, the model is then evaluated using the test set. Because these steps are not in the objective of this post, they are omitted. Given that the model displays an acceptable prediction accuracy, it can then be output and used to make predictions. To output the model so that it can be used in another application, the `pickle` module can be used as follows

```python
import pickle 

#export corpus
with open("corpus.obj", "wb") as corpus_pickle:
    pickle.dump(corpus_vector, corpus_pickle)

#export priors
with open("log_priors.obj", "wb") as priors_pickle:
    pickle.dump(priors_dict, priors_pickle)

#export conditional probabilities
with open("labels.obj", "wb") as labels_pickle:
    pickle.dump(labels_dict, labels_pickle)
```

The code below constructs a quick web application with interactive components that allow a user to input a review and classify the sentiment. A gif of the web application in action is displayed below.

```python
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pickle
import project_utils as nb
import numpy as np


app = dash.Dash(
    external_stylesheets=[dbc.themes.FLATLY],
    prevent_initial_callbacks=True
)

with open("log_Theta_c.obj", "rb") as model_cond_probs:
    log_Theta_c = pickle.load(model_cond_probs)

with open("log_priors.obj", "rb") as model_priors:
    log_priors = pickle.load(model_priors)

with open("labels.obj", "rb") as model_labels:
    labels = pickle.load(model_labels)

with open("corpus.obj", "rb") as model_corpus:
    corpus = pickle.load(model_corpus)

review_input = dbc.Card(
    html.Div(
        children=[
            dbc.Label("Enter your review below!", className="pl-4 pt-1"),
            dbc.Input(
                id="input review", 
                className="mt-2 outline-primary",
            ), 
            dbc.Button("Classify!", id="classify button", color="white", className="btn-outline-success mt-2")
        ], 
        className="border border-primary rounded rounded-5 p-3"
    )
)

review_output = dbc.Card(
    html.Div(
        children=[
            dbc.Label("Classification Results", className="pl-4 pt-1"),
            html.Div(
                id="output review",
                children=[]
            )
        ],
        className="border border-primary rounded rounded-5 p-3"
    )
)

app.layout=(
    dbc.Container(
        [
            html.H1(children="Review Classification", className="text-white bg-primary border border-secondary \
            rounded pl-4 pt-2 pb-2 pr-1 mt-2"), 
            html.Hr(className="bg-primary mb-4 mr-1", style=dict(height="1px")),
            dbc.Row(
                children = [
                    dbc.Col(review_input, md=6), 
                    dbc.Col(review_output,md=6)
                ],
                className="p-1"
            )
        ],
        fluid=True
    )
)

@app.callback(
    Output("output review", "children"), 
    [
        Input("input review", "value"),
        Input("classify button", "n_clicks")
    ],
    prevent_initial_call=True
)

def classify_review(text, clicked):
    text = ["".join(text)]
    if clicked is not None:
        processed_data = nb.process_data(text)
        design_matrix_dict = nb.build_design_matrix(corpus["vector"], processed_data)
        design_matrix = np.array(list(design_matrix_dict.values())).T
        prediction = nb.predict(design_matrix, log_priors["priors"], log_Theta_c)
        probabilities = nb.predict_log_probabilities(design_matrix, log_priors["priors"], log_Theta_c)
        probabilities = np.max(np.exp(probabilities))
        return ("This is a " + str(prediction[0]) + " review" + \
        " with a probability of " + f" {probabilities:.02%}")

if __name__ == "__main__":
    app.run_server(debug=True)
```         

<img src="/assets/img/NaiveBayes/Naive Bayes Prediction.gif" width="100%">.

This sentiment classification application was developed by applying basic NLP concepts on text data combined with the naive Bayes algorithm. Although more complex algorithms and text processing techniques exist to develop significantly more accurate models, applying basic concepts and algorithms can yield sophisticated models which users can rapidly interact with. Hopefully this post demonstrates the value of becoming adept in processing text data and using available resources to create models that can be used to accomplish different tasks. 