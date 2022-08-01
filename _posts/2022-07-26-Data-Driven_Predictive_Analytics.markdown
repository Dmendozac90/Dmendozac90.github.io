---
layout: post
title:  "Dynamic Mode Decomposition Hydrocarbon Predictive Analytics, Anomaly Detection, and Productivity Determination"
date:   2022-07-26 -
description: "This post explores the application of the DMD algorithm on a large hydrocarbon production dataset to demonstrate its potential in time-series predictive analytics, anomaly detection, and productivity quantification. The dataset was obtained from Wattenberg field in Colorado and contains production data form over 4,000 active horizontal wells."
categories: Python Dynamic_Mode_Decomposition Time_Series_Forecast Anomaly_Detection Data-Driven_Methods
html_url: /assets/img/DMDOilProd/DMDProd.webp
---

**Outline**
-   [Introduction](#introduction)
-   [Data Acquisition and Preprocessing](#data_acquisition_and_preprocessing)

-   [DMD Implementation](#dmd_implementation)


## Introduction

Dynamic mode decomposition (DMD) is an equation-free, data-driven method that decomposes high-dimensional data from complex dynamic systems into spatiotemporal structures that can be utilized to make future state predictions and control. The method simply requires snapshots of data from a dynamic system at advancing timesteps. Implementing DMD on a given data set is extremely simple and no assumptions are made about the underlying system, thus making it an extremely popular method to implement within the data science and machine learning communities. Apart from making future state predictions, DMD can also be utilized as a diagnostic tool to characterize different aspects of a system of interest. 

The applications of DMD will be demonstrated on a dataset comprised of monthly hydrocarbon production rates from multi-fractured horizontal wells (MFHW) in Wattenberg field, Colorado. Hydrocarbon production data is readily available for anyone to analyze and thus should be leveraged to draw informative conclusions on various system characterization tasks or other hypotheses of interest. DMD will be utilized to forecast hydrocarbon production volumes and detect production anomalies. Before applying DMD on the production dataset, the data acquisition and preprocessing will be discussed briefly. 

## Data Acquisition and Preprocessing

The production dataset was generated using **[Selenium WebDriver](https://www.selenium.dev/documentation/webdriver/)**  to obtain the monthly production volumes reported on the Colorado Oil and Gas Conservation Commission  **[(COGCC)](https://cogcc.state.co.us/data.html)** website across thirty-eight (38) township and ranges within Wattenberg field. Only active horizontal wells were considered in this analysis and are displayed below. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/DMDOilProd/Wattenberg Horizontal Well Location.html" height="525" width="100%"></iframe>

To assess the accuracy of the DMD-derived production forecasts, the data must be divided into a training and test set. The training set will include the first fifty (50) months of production and thus wells that do not meet this requirement will be omitted. Furthermore, wells that have empty production values within the training window are also omitted from the training dataset. This decision was made because the objective of this analysis attempts to characterize the natural productivity of MFHWs completed in Wattenberg field and thus including production that has been modified due to operational constraints will not accurately reflect natural productivity. An example of empty production values within the training window is demonstrated below.

<img src="/assets/img/DMDOilProd/NaNprod.PNG" width="100%">

Notice that this well has empty production on the fifth month. Applying DMD on such data will modify the spatiotemporal structures to include this as natural flow behavior of MFHWs which is incorrect. Note the low volume listed in the first month of production. Because wells are not necessarily completed at the beginning of a month, it is often that the first month of production contains volumes associated with a number of days less than a month. For this reason, all first month production values were not included in the training dataset. This means that this DMD model will only be applicable if the first month production value corresponds to a well producing for a full month or at the second month of production. After the training dataset was finalized, DMD was applied on the production dataset.

## DMD Implementation

To apply DMD on the production dataset, a function was defined to implement the algorithm. The following code block carries out the DMD algorithm. 

```python
def DMD_fit(X_train, X_2, U, Sigma, V, rank, r):
    """
    parameters
    ----------

    X_train : training dataset
    X_2 : shifted dataset
    U : left-singular vectors
    Sigma : singular values
    V : right-singular vectors
    rank : optimal rank
    r : right pointer 
    """ 
    U, Sigma, V = U[:, :rank], Sigma[:rank], V[:rank, :]
    A_tilde = np.linalg.multi_dot([U.T, X_2, V.T, np.diag(np.reciprocal(Sigma))])
    Lambda, W = np.linalg.eig(A_tilde)
    Phi = np.linalg.multi_dot([X_2, V.T, np.diag(np.reciprocal(Sigma)), W])
    b, residuals, rank, sigma = np.linalg.lstsq(Phi, X_train[:, 0], rcond=None)
    temp = np.repeat(Lambda.reshape(-1,1), (r - 1), axis=1)
    tpower = np.arange(r - 1)
    dynamics = np.power(temp, tpower) * b[:r].reshape(rank, -1)
    X_dmd = Phi @ dynamics
    
    #return reconstructed data, eigenvalues, modes, and initial condition
    return X_dmd, Lambda, Phi, b
```

The function can then be called to obtain the spatiotemporal structures that define the model.

```python
X_1 = X_train[:, :-1]
X_2 = X_train[:, 1:]
U, Sigma, V = np.linalg.svd(X_1)
V = V.conj()
X_dmd, Lambda, Phi, b = DMD_fit(X_train, X_2, U, Sigma, V, 12, 50)
```

To obtain the appropriate value for the rank, one must evaluate the error using a loss metric on a validation set. The following graph displays the training and validation errors using the root mean squared loss.

<img src="/assets/img/DMDOilProd/Training_Test_Error.webp" width="100%">

These results indicate that the appropriate rank to utilize for the model is eleven (11). With the DMD modes, eigenvalues and initial conditions computed, future state predictions can be made. The following function forecasts production for the wells included in the training window.

```python
def DMD_forecast(Lambda, Phi, b, r, timesteps,):
    tpower = np.arange(r - 1, r + timesteps - 1)
    temp = np.repeat(Lambda.reshape(-1, 1), tpower.shape[0], axis=1)
    dynamics = np.power(temp, tpower) * b.reshape(Lambda.shape[0], -1)
    X_forecast = Phi @ dynamics

    return X_forecast
```

The image below displays a sample obtained from the DMD algorithm. 

<img src="/assets/img/DMDOilProd/DMDProd2.webp" width="100%">

The purple line displays the DMD-reconstructed data within the specified training window. The orange line displays the DMD-derived production forecasts. Note the accurate predictions that are made by the model on data outside training window. With these simple functions, every well included in the training dataset now has future state predictions. A great way to utilize these forecasts is to detect significant deviations in well productivity as it occurs in real time. Doing so quickly notifies engineers if a particular set of wells require immediate intervention to fix losses in production volumes or if a recently completed production workover augments previous production volumes. The following function detects such productivity deviations.

```python
def anomaly_detect(error, keys, tol, seq):
    anomalous_plus, anomalous_neg = [], []
    for i, e in enumerate(error):
        try:
            tol_array_plus = e >= tol
            tol_array_neg = e <= -tol
            plus_seq = np.diff(np.where(np.concatenate(([tol_array_plus[0]], 
                    tol_array_plus[1:] != tol_array_plus[:-1], [True])))[0])[::2]
            neg_seq = np.diff(np.where(np.concatenate(([tol_array_neg[0]], 
                    tol_array_neg[1:] != tol_array_neg[:-1], [True])))[0])[::2]
            if np.sum(plus_seq) >= seq:
                anomalous_plus.append((keys[i], i))
            if np.sum(neg_seq) >= seq:
                anomalous_neg.append((keys[i], i))
        except: Exception
    
    return anomalous_plus, anomalous_neg
```

The image below displays an example of a well that begins to deviate negatively significantly. 

<img src="/assets/img/DMDOilProd/NegProd.webp" width="100%">

Detecting these deviations quickly and in an automated manner is of great benefit because it allows for the implementation of intervention methods quickly and efficiently. Positive deviations can also be utilized to determine the augment in production volumes related to implementing production enhancement techniques like artificial lift. It may also serve as a tool to detect wells that may have been placed under artificial lift. The image below displays a well with positive production deviations.

<img src="/assets/img/DMDOilProd/PosProd.webp" width="100%">

It is evident that action was taken to augment the declining production of this well. This is just one example of how the diagnostic properties of the DMD algorithm can be leveraged to develop clever and innovative solutions utilizing available data. With just a few lines of code, a robust and automated method for forecasting production, detecting unexpected declines in production, and demonstrating the augment in production was quickly implemented. Another potential use for the DMD algorithm could take advantage of the eigenvalues that are calculated to determine productivity of a particular area.

The eigenvalues describe the growth/decay and oscillatory behavior of the spatiotemporal modes. By computing the eigenvalues associated with each township/range in Wattenberg field, it may demonstrate the variation in productivity. This is of particular interest because it may help demonstrate areas of higher productivity and thus allow for focus to be placed in regions with more hydrocarbon productivity potential. The simplicity of implementing this algorithm and its wide array of applications is remarkable thus making the DMD algorithm a great tool within data science.