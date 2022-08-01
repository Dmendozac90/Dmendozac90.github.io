---
layout: post
title:  "Predict User Advertisement Click Behavior"
date:   2022-02-03 -
description: "In this post, a dataset comprised of numeric and categorical features will be analyzed to determine if it is possible to predict whether an individiual is likekly to click on an online advertisement."
categories: Python Logistic_Regression Classification Regularized_Discriminant_Analysis Feature_Importance 
html_url: /assets/img/PredictAdClick/Violin_plots.webp
---

**Outline**
-   [Introduction](#introduction)
-   [Dataset Exploration and Visualization](#dataset-exploration-and-visualization)
-   [Data Preprocessing and Model Investigation](#data-preprocessing-and-model-investigation)
-   [Feature Importance](#feature-importance)
-   [Conclusion](#conclusion)


## Introduction

The ability to predict the behavior of an individual provides valuable information for advertising companies. Knowing whether an individual is likely to click on an online advertisement allows for advertisements to be distributed in a way such that the target population will provide a positive response to a particular advertisement. Furthermore, understanding which features provide the most information to generate an accurate prediction can be used to reduce costs or make informative data acquisition modifications. This post will cover an analysis of a marketing dataset that consists of multiple numerical and categorical features with the objective to predict if a user will click or not click on an advertisement. 

## Dataset Exploration and Visualization

The primary step in any machine learning task is to analyze the data. Every task will have unique datasets with varying features and data types and one should become quickly acquainted with the dataset to determine possible preprocessing and modeling procedures that will need to be implemented. After the dataset is loaded as a pandas `DataFrame` object, the `head` and `info` methods can be used to quickly observe the data and examine the summary of the data. In this analysis, the dataset is stored in the variable `data`.

```python
data.head()
```

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/PredictAdClick/dataframe_head.html" height="350" width="100%"></iframe>

```python
data.info()

<class pandas.core.frame.DataFrame>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 10 columns):
 #   Column                    Non-Null Count  Dtype  
"---  ------                    --------------  -----"  
 0   Daily Time Spent on Site  1000 non-null   float64
 1   Age                       1000 non-null   int64  
 2   Area Income               1000 non-null   float64
 3   Daily Internet Usage      1000 non-null   float64
 4   Ad Topic Line             1000 non-null   object 
 5   City                      1000 non-null   object 
 6   Gender                    1000 non-null   int64  
 7   Country                   1000 non-null   object 
 8   Timestamp                 1000 non-null   object 
 9   Clicked on Ad             1000 non-null   int64  
dtypes: float64(3), int64(3), object(4)
memory usage: 78.2+ KB
```

The `info` methods reveals that the data contains no missing data. The "Non-Null Count" indicates that all entries in the respective column contain data types as shown by "Dtype". It is also shown that the dataset contains $$6$$ numeric features and $$4$$ objects. The unique values in the numeric features can then be inspected to determine if any features are categorical.

```python
columns = data.columns
for col in columns:
    if data[col].dtype != object:
        unique = len(data[col].unique())
        print("There are " f'{unique}' " unique elements in the attribute " f'{col}')

"There are 900 unique elements in the attribute Daily Time Spent on Site
There are 43 unique elements in the attribute Age
There are 1000 unique elements in the attribute Area Income
There are 966 unique elements in the attribute Daily Internet Usage
There are 2 unique elements in the attribute Gender
There are 2 unique elements in the attribute Clicked on Ad"
```

It can be seen that "Gender" and "Clicked on Ad" encode a binary categorical attribute/response given that there are only two unique numbers, namely $$0$$ and $$1$$ in both columns. This data has been preprocessed; however, this may not always be the scenario that is encountered. If the binary attribute were to be denoted by text, the data would require to be encoded numerically before it can be modeled using machine learning models. The same step is conducted on the object data types in the dataset. 

```python
columns = data.columns
for col in columns:
    if data[col].dtype == object:
        unique = len(data[col].unique())
        print("There are " f'{unique}' " unique elements in the attribute " f'{col}')

"There are 1000 unique elements in the attribute Ad Topic Line
There are 969 unique elements in the attribute City
There are 237 unique elements in the attribute Country
There are 1000 unique elements in the attribute Timestamp"
```

From these results, one may conclude that "City" and "Country" will likely not contribute to improving the accuracy of classification model. Given that these two attributes contain $$969$$ and $$237$$ unique cities and countries, and that the dataset consists of $$1,000$$ instances, any groupings in cities and countries will be too small and thus will not contribute significantly to determine if a specific user clicked or did not click on an advertisement. Although one may be inclined to make the same conclusion about the "Ad Topic Line" and "Timestamp" attributes, it would be incorrect to assume this because these attributes require further analysis to determine if there is any correlation between these two attributes and the response feature. One may consider searching for a set of words in the "Ad Topic Line" that may be indicative of whether an individual is prone to clicking an advertisement. The timestamp data can also be grouped into larger time intervals and analyze if individuals are prone to clicking an advertisement during a particular time. 

Before these attributes are analyzed, focus is placed on the numeric features. A summary of the descriptive statistics is provided along with the correlation coefficients using the pandas `describe` and `corr` methods. 

```python
numeric_data.describe()
```

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/PredictAdClick/Summary_Stats.html" height="250" width="100%"></iframe>

```python
corr_coeff = numeric_data.corr()
```

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/PredictAdClick/Corr_coeff.html" height="175" width="100%"></iframe>

No significant results are observed from either output. The data can next be explored using visualizations and observe if there are any significant features that may relate the attributes to the response variable. Histograms and scatter plots are two simple and effective visualizations that can outline significant features and aid in further understanding the given dataset. Pandas provides a method for creating a matrix of scatter plots which display both histograms and scatter plots for the specified numeric features. Although this method is sufficient to create an effective data visualization, plotly's graphing libraries provide the tools to construct interactive plots that can be highly beneficial while exploring a dataset and thus will be used instead of the standard plotting libraries. The scatter plot matrix is displayed below. 

<iframe id="igraph" scrolling="yes" style="border:none;" seamless="seamless" src="/assets/img/PredictAdClick/Scatterplot_matrix.html" height="825" width="100%"></iframe>

The data can be examined by hovering over any graph of interest. This triggers a display of a hover label that displays information about the data contained in each plot. The histograms display the corresponding bin ranges and instance counts within those bin ranges. Additionally, the calculated kernel density estimate (KDE) is displayed for each histogram plot. The scatter plots display each feature plotted against another numeric feature. The column names are displayed along the x-axis and the hover label displays the features that are plotted on the x and y axes, respectively. 

The histograms for the "Daily Time Spent on Site" and "Daily Internet Usage" display features associated with a bimodal distribution. This suggest that there are two underlying groups in the data. Also, if the scatter plots are examined carefully, the concentration of data appears to also be concentrated in two groups. This is especially evident with the scatter plot that displays "Daily Time Spent on Site" and "Daily Internet Usage". Given that the numeric data may adequately describe the binary response variable, it may serve to analyze the scatterplot matrix by coloring the data according to the response variable. The following scatter plot demonstrates these results. 

<iframe id="igraph" scrolling="yes" style="border:none;" seamless="seamless" src="/assets/img/PredictAdClick/Scatterplot_matrix_color.html" height="825" width="100%"></iframe>

It is easily observed that there is a direct correlation amongst the numeric features of the data and the response variable given the clear grouping of the data when colored. Note that the bimodal distribution of the histograms also appears to be generated by the grouping of the response variable. To further demonstrate the correlation, the data is colored on the gender of the individuals and displayed below.

<iframe id="igraph" scrolling="yes" style="border:none;" seamless="seamless" src="/assets/img/PredictAdClick/Scatterplot_matrix_color_2.html" height="825" width="100%"></iframe>

Note that the same separation is not observed in this scatter plot matrix. This can be investigated by toggling on and off the gender category by clicking on the entry on the legend of the plot. When either gender is removed from the scatter plot, one can observe that both genders are distributed similarly. Given that a clear pattern has been observed from the scatter plot matrices, it must be analyzed thoroughly. 

The data is grouped according to the response variable and then the histograms for the attributes "Daily Time Spent on Site" and "Daily Internet Usage" are reanalyzed. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/PredictAdClick/Filtered_histograms.html" height="525" width="100%"></iframe>

The histograms display two distinct normal distributions and thus confirming that there are groups described by these attributes. Note that individuals that clicked on an advertisement are primarily characterized by spending lower time on the site and on the internet on a daily basis. A great visualization tool to display the degree of the overlap of the normal distributions are violin plots. Violin plots display the KDE and the box and whisker plot that describes the data. The first and second standard deviations have been added to demonstrate the degree of overlap. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/PredictAdClick/Violin_plots.html" height="525" width="100%"></iframe>

Note that the overlap occurs after one standard deviation from both datasets. Given the information observed from the visualization of the numerical data, it may suggest that these attributes are sufficient to develop a model that can accurately classify the click behavior of an individual. The data will then be processed for modeling

## Data Preprocessing and Model Investigation

Because the numeric features appear to be sufficient for the classification task, only this data will be preprocessed and used to generate classification models. If the model evaluation suggest that the numeric features are not sufficient to generate an accurate model, then the time and advertisement topic data will be processed to augment model accuracy. Recall that this model-generating process corresponds to generating an accurate classification model. If the objective included an investigation of how all the provided features relate to advertisement response, the text and time data would require a thorough analysis during the data exploration and visualization stage. 

The first step is to create a training and test set. If the proportions of classes in the dataset are not even, generating the training and test sets randomly will incur a sampling bias and thus affect the accuracy of the model. To avoid this unwanted effect, the training and test sets will be created using stratified sampling. This will ensure that the proportions of the classes in the dataset will remain approximately equal when the data is split. The `stratified_shuffle_split` method from scikit learn is used to generate stratified training and test datasets. 

```python
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(data.iloc[:, :-1].to_numpy(), data.iloc[:, -1].to_numpy()):
    data_train = data.loc[train_index]
    data_test = data.loc[test_index]
#Create trainging and test dataset numpy arrays
train_array, test_array = data_train.to_numpy(), data_test.to_numpy()
```
Because the numeric features are described by different scales, the data must be standardized. This is done using scikit learn's `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
#training array is sliced to incorporate continuous numeric features
scale.fit(train_array[:,:4])
#training and test sets are scaled and then combined with the gender numeric attribute
X_train = np.c_[scale.transform(train_array[:, :4]).astype("float"), 
                train_array[:, 6:7].astype("float")]
X_test = np.c_[scale.transform(test_array[:, :4]).astype("float"), 
               test_array[:, 6:7].astype("float")]
y_train = train_array[:, -1].astype("int")
y_test = test_array[:, -1].astype("int")
```

Now that the numeric data has been preprocessed, models can be generated and assessed. Two different classifiers will be investigated: logistic regression and linear discriminant analysis. The classifiers are first imported.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

Initial model parameters are specified and then the models are trained on the training dataset.

```python
log_clf = LogisticRegression(
    penalty="l1",
    tol=1e-4,
    C=1e6,
    solver="liblinear",
)

lda_clf = LinearDiscriminantAnalysis(
    solver="lsqr", 
    shrinkage=1e-6
)

log_clf.fit(X_train, y_train)
lda_clf.fit(X_train, y_train)
```

The predictive accuracy of each model is evaluated using subsets of the training data called validation data. Scikit learn provides a useful feature that further splits the training data into another training dataset and a validation dataset and then calculates the accuracy of the model on the unseen validation data. 

```python
from sklearn.model_selection import cross_val_score
cross_val_score(log_clf, X_train, y_train, cv=5, scoring="accuracy"),\
np.mean(cross_val_score(log_clf, X_train, y_train, cv=5, scoring="accuracy"))

(array([0.93125, 0.975  , 0.96875, 0.98125, 0.96875]), 0.9650000000000001)

cross_val_score(lda_clf, X_train, y_train, cv=5, scoring="accuracy"),\
np.mean(cross_val_score(lda_clf, X_train, y_train, cv=5, scoring="accuracy"))

(array([0.94375, 0.96875, 0.95   , 0.98125, 0.95625]), 0.96)
```

The logistic regression and linear discriminant analysis models display a high degree of accuracy. As suspected from the scatterplot matrices, the numeric data contains sufficient information that can describe whether an individual will click on an advertisement or not. Although it may appear that the main objective has been satisfied, it is encouraged that model regularization be investigated. Because the model has been trained on limited data, the accuracy of this model may begin to decline as more data is acquired and thus altering the original observed distribution. Model regularization limits model overfitting and thus enhances the generalization predictive ability of the model. Regularization is investigated using scikit learn's `GridSearchCV`. 

```python
from sklearn.model_selection import GridSearchCV

grid = [
    dict(C=list(range(1, 11)))
]

grid_search = GridSearchCV(log_clf, grid, cv=5,
 scoring="accuracy")

 grid_search.fit(X_train, y_train)

 grid_search.best_estimator_,  grid_search.best_score_

 LogisticRegression(C=4, penalty='l1', solver='liblinear'),  0.9662499999999999
```

The model accuracy is marginally improved but this regularized logistic regression model may be able to generalize better and thus will be selected. The same procedure is conducted for the LDA model except the shrinkage parameter is modified.

```python
grid = [
    dict(shrinkage=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
]

grid_search = GridSearchCV(lda_clf, grid, cv=5,
 scoring="accuracy")

grid_search.fit(X_train, y_train)

 grid_search.best_estimator_,  grid_search.best_score_

(LinearDiscriminantAnalysis(shrinkage=0.2, solver='lsqr'), 0.96)
```

The LDA model accuracy is not improved and is less than logistic regression and thus the logistic model will be selected. The logistic regression model is revaluated by generating a confusion matrix. This matrix displays the ability of the model to accurately predict the true class labels and also displays which labels it is classifying incorrectly.

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, log_clf.predict(X_test))

array([[99,  1],
       [ 5, 95]], dtype=int64)
```

This matrix displays that the logistic regression model correctly predicts $$99$$ of the $$100$$ individuals who are classified as those that did not click the advertisement on the test dataset. It also correctly predicts $$95$$ of the $$100$$ individuals that did click on the advertisement. These are very accurate results and thus it may be that our model is sufficiently accurate to be used to predict user click behavior. 

An interesting result from this analysis was that not all of the attributes of the data were utilized to generate a sufficiently accurate model. This observation may also suggest that all the numeric features may not be equally important. The relevance of each feature may also provide valuable information and thus should be considered in this analysis. This can be investigated using l1-regularized logistic regression. This analysis is shown in the following section.

## Feature Importance

L1-regularization creates sparse models that phase out attributes that do not significantly contribute to the model as regularization is increased. Once a parameter is set to $$0$$ from the increased regularization, it will no longer become active. Parameter values can be tracked as regularization is increased and thus the path of the coefficients can be displayed graphically. The l1-regularized path of the coefficients for this model are displayed below. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/PredictAdClick/Log_reg_coeffs.html" height="525" width="100%"></iframe>

Note that the first coefficient that goes to $$0$$ corresponds to the "Gender" attribute. This coefficient therefore contributes the least for the model. Although it was not evident from the scatter plot matrix when it was colored on gender, the lack of correlation observed did suggest this would be the case. Given that the correlation in the scatter plot matrix is indicative of how well it contributes to the model, "Daily Time Spent on Site" and "Daily Internet Usage" should not be phased out unless the model is highly regularized. This is exactly what is displayed in figure. The next coefficients that become $$0$$ are "Age" and "Area Income". Although they approach $$0$$  approximately equally, "Age" appears to reach $$0$$ first. As expected from the scatter plot matrix, "Daily Internet Usage" and "Daily Time Spent on Site" are the last coefficients that become $$0$$. From this analysis, it can be determined that the coefficients can be ranked as follows:

**1)** *Daily Internet Usage*

**2)** *Daily Time Spent on Site*

**3)** *Area Income*

**4)** *Age*

**5)** *Gender*

Another model that generates feature importance is a random forest classifier. If a model is trained using scikit learn's `RandomForrestClassifier`, the importance of the features can be displayed. This is done as follows:

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, y_train)
rnd_clf.feature_importances_

#[Daily Time Spent on Site, Age, Area Income, Daily Internet Usage, Internet]
array([0.34522895, 0.08892486, 0.11168584, 0.44937014, 0.00479021])
```

These results correlate to the order that was identified using the l1-regularized logistic regression model. This information is valuable as it may be used to modify the data acquisition. 

## Conclusion

The analysis of this marketing dataset revealed that not all the provided data attributes were required to create a highly accurate logistic regression classifier. Given that this was the main objective in this analysis, all of the data attributes were not analyzed. This methodology may be appropriate given the objective and any associated time constraints; however, if time allows, one should examine the dataset thoroughly as it may uncover key insights that will augment the information provided by the analysis. 

The insights derived from this dataset were originally observed by generating simple data visualizations. The patterns discovered from these visuals directed the analysis towards limiting the dataset to the numerical features. These features were demonstrated to be sufficient to generate a highly accurate logistic regression classifier. Given that all the features were not necessary, it was natural to question how each numerical feature contributed towards the final prediction of the classification model. Using 1-regularization, the contribution of each numeric feature was determined, and a comprehensive ranking of these features was established. These results were correlated with the ranking provided by a random forest classifier. 

These results may be used to predict if an individual is likely to click on an advertisement, develop a strategy towards targeting those who will be perceptive towards online advertising, and make decisions if data acquisition should be modified given that not all features are necessary and not all features contribute significantly towards the accuracy of the model. 

