---
layout: post
title:  "A Probabilistic Approach to Linear Regression"
date:   2022-01-03 -
description: "Linear Regression is a core method used in statistics and machine learning to construct models that can be used for quantitative prediction and parameter inference. The method of least squares is typically used to estimate the model parameters and is implemented by minimizing the residual sum of squares between the model's predicted values and the true values. Although minimizing the residual sum of squares is easily comprehended, its statistical derivation is not often well understood. Understanding the statistical perspective of linear regression is fundamental for understanding model regularization and further topics in machine learning. The statistical perspective of linear regression is explored and implemented on real data."
categories: Python Linear_Regression Regularization 
html_url: /assets/img/LinearRegression/lstsq.html
---

**Outline**
-   [Introduction](#introduction)
-   [Linear Regression](#linear-regression)
-   [Probabilistic Derivations of Regularized Linear Models](#probabilistic-derivations-of-regularized-linear-models)
-   [Linear Regression on Prostate Dataset](#linear_regression_on_prostate_dataset)

## Introduction

Probability theory provides engineers, scientists, and other professionals from a wide array of technical disciplines with a set of tools that can be utilized to analyze complex problems that involve uncertainty. The abundance of data and exponential growth of computational power have led to a rise in demand for automated analytical methods that provide valuable insights distilled from data. Automated analytical methods are found within the scope of machine learning. Although these analytical methods are based on well-understood statistical models and algorithms, the results provided by these methods always include an associated degree of uncertainty and thus are well-described within the framework of probability theory. 

Machine learning often introduces linear models early because their simple implementation and interpretability thus provide a solid foundation to learn introductory machine learning principles. Linear models are often introduced within the framework of linear algebra. The model is usually presented in the form of $$\mathbf{Ax} = \mathbf{b}$$ where the objective is to obtain a solution for $$\mathbf{x}$$. Generally, a solution for $$\mathbf{x}$$ does not exist and thus the best-approximation for $$\mathbf{x}$$ is computed in a *least-square* sense. 

Because linear algebra provides an elegant mathematical and geometric explanation for the optimization process of estimating $$\mathbf{x}$$, novice machine learning practitioners may not consider the probabilistic interpretation of linear regression. This may create issues when more advanced topics in machine learning are encountered as they require understanding of topics in probability theory. Given the simplicity of linear regression models, it is important to develop an understanding of the probabilistic framework by applying these topics on linear regression.

This post will begin by presenting a thorough outline of linear regression within the scope of linear algebra. The alternative perspective derived within a probabilistic framework will then be introduced, followed by two common methods of regularization and their implementation in python. Lastly, an example of linear regression will be applied on a dataset to compare the models presented in this post.

## Linear Regression

In machine learning, regression tasks construct a mapping that transform a set of inputs to real-valued outputs. The inputs and outputs form a training dataset denoted as $$\mathcal{D} = \{(\mathbf{x}_i , y_i)\}^N_{i=1}$$ where $$\mathbf{x}_i$$ is a $$D-$$dimensional feature vector of inputs, $$y_i$$ is the corresponding output, and $$N$$ is the number of samples in the dataset. Assuming that the inputs and outputs are linearly related, a linear regression model is used to map the inputs to the output values. The linear regression model is written in matrix form as 

$$
\mathbf{y} = \mathbf{Xw}
$$

Although the bias term is not explicitly denoted in this equation, it's included by using a common notational trick that one should familiarize themselves with. The bias is included by prepending a vector of $$N$$ ones in the data matrix $$\mathbf{X}$$. To demonstrate this, consider a row of $$\mathbf{X}$$. The expansion of the inner product of a row $$\mathbf{x}_i$$ and the weight vector $$\mathbf{w}$$ can be written as 

$$
\mathbf{w}^{T}\mathbf{x}_i = \left[\begin{array}{c}w_0 & w_1 & \cdots & w_d \\ \end{array} \right]\left[\begin{array}{c}1 \\ x_1 \\ \vdots \\ x_D  \end{array} \right] 
$$

$$
= w_0 + w_{1}x_{1} + \cdots + w_{d}x_{d}
$$

Because $$w_0$$ is multiplied by $$1$$ and then summed with the remaining $$D$$ elements, the bias term gets included simply by prepending $$1$$ onto every row of the matrix $$\mathbf{X}$$. The training set $$\mathcal{D}$$ is used to estimate the weight vector $$\mathbf{w}$$. The most common method to approximate weight vector $$\mathbf{w}$$ is by the method of *least squares*. The objective of this method is to minimize the residual sum of squares by selecting the optimal coefficients of the weight vector. The residual sum of squares is 

$$
RSS(\mathbf{w}) = \sum_{i=1}^{N}(y_{i} - \mathbf{w^{T}x_{i}})^2
$$

and in matrix form

$$
RSS(\mathbf{w}) = (\mathbf{y} - \mathbf{Xw})^{T}(\mathbf{y} - \mathbf{Xw})
$$

Note that the $$RSS(\mathbf{w})$$ is a quadratic function with respect to $$\mathbf{w}$$ and thus a global minimum exists. To find the minimum the derivative of $$RSS(\mathbf{w})$$ with respect to $$\mathbf{w}$$ must be determined. The differentiation of $$RSS(\mathbf{w})$$ is as follows

$$
\frac{\partial}{\partial \mathbf{w}}RSS(\mathbf{w}) =\frac{\partial}{\partial \mathbf{w}} (\mathbf{y} - \mathbf{Xw})^{T}(\mathbf{y} - \mathbf{Xw})
$$

$$
 = \frac{\partial}{\partial \mathbf{w}}(\mathbf{y}^{T} \mathbf{y} - \mathbf{y}^{T}\mathbf{Xw} - \mathbf{w}^{T} \mathbf{X}^T \mathbf{y} + \mathbf{w}^{T} \mathbf{X}^T \mathbf{Xw})
$$

Although it may not be apparent, the two inner terms are equivalent and can be combined. To prove this, the associative law of matrix multiplication will be used. The associative law is stated as

$$
\mathbf{A}(\mathbf{BC}) = (\mathbf{AB})\mathbf{C}
$$

Thus the order of matrix multiplication is not important and if $$\mathbf{y}^{T}\mathbf{Xw}$$ is expanded by first multiplying $$\mathbf{Xw}$$, the following is obtained

$$
\mathbf{y}^{T}(\mathbf{Xw}) = \left[\begin{array}{c}y_1 & y_2 & \cdots & y_N \\ \end{array} \right] (\left[\begin{array}{c} 1 & x_{1, 1} & \cdots & x_{1, D} \\ \vdots & \vdots & \ddots & \vdots \\  1 & x_{N, 1} & \cdots & x_{N, D} \end{array} \right] \left[\begin{array}{c} w_0 \\ w_1 \\ \vdots \\ w_D \end{array} \right])
$$

$$
= y_{1}(w_{0} + w_{1}x_{1,1} + \cdots + w_{D}x_{1,D}) + \cdots + y_{N}(w_{0} + w_{1}x_{N,1} + \cdots + w_{D}x_{N,D})
$$

Expansion of the second inner term yields

$$
\mathbf{w}^{T} \mathbf{X}^T \mathbf{y} = \left[\begin{array}{c} w_{0} & w_{1} & \cdots & w_{D} \\ \end{array} \right] \left[\begin{array}{c} 1 & \cdots & 1 \\ x_{1,1} & \cdots & x_{1,N} \\ \vdots & \ddots  & \vdots \\ x_{D, 1} & \cdots & x_{D,N}\end{array} \right] \left[\begin{array}{c} y_1 \\ y_2 \\ \vdots \\ y_N \end{array} \right]
$$

$$
= y_{1}(w_{0} + w_{1}x_{1,1} + \cdots + w_{D}x_{D,1}) + \cdots + y_{N}(w_{0} + w_{1}x_{1,N} + \cdots + w_{D}x_{D,N})
$$

The two expansions are equivalent and thus $$\mathbf{y}^{T}\mathbf{Xw}$$ and $$\mathbf{w}^{T} \mathbf{X}^T \mathbf{y}$$ can be combined 

$$
 \frac{\partial}{\partial \mathbf{w}}(\mathbf{y}^{T} \mathbf{y} - \mathbf{y}^{T}\mathbf{Xw} - \mathbf{w}^{T} \mathbf{X}^T \mathbf{y} + \mathbf{w}^{T} \mathbf{X}^T \mathbf{Xw})
$$

$$
= \frac{\partial}{\partial \mathbf{w}}(\mathbf{y}^{T} \mathbf{y}  - 2\mathbf{w}^{T} \mathbf{X}^T \mathbf{y} + \mathbf{w}^{T} \mathbf{X}^T \mathbf{Xw}) 
$$

Thus

$$
\frac{\partial}{\partial \mathbf{w}}RSS(\mathbf{w}) = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^{T}\mathbf{X}\mathbf{w}
$$

The next step is to equate the expression above to $$0$$ and solve for $$\mathbf{w}$$ 

$$
-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^{T}\mathbf{X}\mathbf{w} = 0
$$

$$
\mathbf{X}^{T}\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}
$$

$$
\mathbf{w} = (\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}
$$

This analytical solution for $$\mathbf{w}$$ corresponds to the solution that minimizes the squared difference between $$y_i$$ and $$\mathbf{w}^{T}\mathbf{x}_{i}$$. The mathematical intuition behind this analytical derivation is easy to grasp. This solution can also be demonstrated visually by providing the geometric interpretation of this analytical solution. To demonstrate this, consider the following example where the matrix $$\mathbf{X}$$ is defined as 

$$
\mathbf{X} = \left[\begin{array}{c}2 & 0 \\ 2 & 2 \\  2 & 4 \end{array} \right]
$$

The number of samples $$N$$ is $$3$$ and the number of features $$D$$ is $$2$$. The output vector $$\mathbf{y}$$ is 

$$
\mathbf{y} = \left[\begin{array}{c} 6 \\ 0 \\ 0 \end{array} \right]
$$

The columns of $$\mathbf{X}$$ define a linear subspace for which all solutions to the linear system $$\mathbf{y} = \mathbf{Xw}$$ exist. The set of possible solutions correspond to a plane within $$\mathbb{R}^{N}$$ and is displayed in the figure below. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/geo_intrep.html" height="525" width="100%"></iframe>

The first two entries in the legend **x_1** and **x_2** are the columns vectors of $$\mathbf{X}$$ and **y** is the output vector $$\mathbf{y}$$. Thus

$$
\mathbf{x}_1 = \left[\begin{array}{c} 2 \\ 2 \\ 2 \end{array} \right] \mathbf{x}_2 = \left[\begin{array}{c} 0 \\ 2 \\ 4 \end{array} \right]  \mathbf{y} = \left[\begin{array}{c} 6 \\ 0 \\ 0 \end{array} \right] 
$$

Note that **x_1** and **x_2** lie along the plane in the figure but the vector $$\mathbf{y}$$ does not. This implies that a solution for $$\mathbf{y} = \mathbf{Xw}$$ does not exist. Recall that $$\mathbf{w} = (\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}$$ provides a solution that minimizes the sum of the squared differences between $$\mathbf{y}$$ and $$\mathbf{Xw}$$. If this solution is substituted into $$\mathbf{y} = \mathbf{Xw}$$ the following expression is obtained

$$
\hat{\mathbf{y}} = \mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}
$$

The vector $$\hat{\mathbf{y}}$$ is calculated in python as

```python
X = np.array([[2, 0], [2, 2], [2, 4]])
y = np.array([[6], [0], [0]])

y_hat = X @ np.linalg.inv(X.T @ X) @ X.T @ y
print(y_hat)

[[ 5.]
 [ 2.]
 [-1.]]
```

The vector labeled **y_hat** in the figure above displays this vector. Note that this vector lies in the column space of $$\mathbf{X}$$ thus $$\hat{\mathbf{y}} \in span(\mathbf{X})$$. The residual vector $$\mathbf{y} - \hat{\mathbf{y}}$$ is also displayed and corresponds to the vector **y - y_hat**. The solution $$\mathbf{w} = (\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}$$ ensures that the residual vector $$\mathbf{y} - \hat{\mathbf{y}}$$ is minimized thus the residual vector must be orthogonal to the column space of $$\mathbf{X}$$. This can be verified by computing the inner product of the residual vector and any vector in the column space of $$\mathbf{X}$$.

$$
(\mathbf{y} - \hat{\mathbf{y}})^{T} \hat{\mathbf{y}} = \left[\begin{array}{c} 1 & -2 & 1 \end{array} \right]\left[\begin{array}{c} 5 \\ 2 \\  -1 \end{array} \right]
$$

$$
= (1 \times 5)+ (-2 \times 2) + (1 \times -1) = 0
$$

$$
(\mathbf{y} - \hat{\mathbf{y}})^{T} \mathbf{x}_1 = \left[\begin{array}{c} 1 & -2 & 1 \end{array} \right]\left[\begin{array}{c} 2 \\ 2 \\ 2 \end{array} \right]
$$

$$
= (1 \times 2)+ (-2 \times 2) + (1 \times 2) = 0
$$

$$
(\mathbf{y} - \hat{\mathbf{y}})^{T} \mathbf{x}_2 = \left[\begin{array}{c} 1 & -2 & 1 \end{array} \right]\left[\begin{array}{c} 0 \\ 2 \\ 4 \end{array} \right]
$$

$$
= (1 \times 0)+ (-2 \times 2) + (1 \times 4) = 0
$$

When the operation $$\hat{\mathbf{y}} = \mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}$$ is carried out, the matrix $$\mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}$$ projects $$\mathbf{y}$$ orthogonally onto the column space of $$\mathbf{X}$$. Another method that verifies that the orthogonal projection of $$\mathbf{y}$$ onto the column space of $$\mathbf{X}$$ minimizes the residual vector is to compute the euclidean norm of the residual vector and compare it to other residual vectors computed from vectors that lie in the columns space of $$\mathbf{X}$$. The euclidean norm of $$\mathbf{y} - \hat{\mathbf{y}}$$ is computed as follows

$$
\begin{Vmatrix}
\mathbf{y} - \hat{\mathbf{y}}
\end{Vmatrix}_2  = \sqrt{(1^{2} + -2^{2} + 1^{2})} \approx 2.45
$$

Next, an arbitrary vector that lies in the column space of $$\mathbf{X}$$ is selected. This vector is

$$
\mathbf{x}_3 = \left[\begin{array}{c} 4 \\ 0 \\ -4 \end{array} \right]
$$

and is also included in the figure above as **x_3** along with its residual vector **y - x_3**. The norm of the residual vector is then computed 

$$
\begin{Vmatrix}
\mathbf{y} - \mathbf{x}_3
\end{Vmatrix}_2 = \sqrt{(2^{2} + 0^{2} + 4^{2})} \approx 4.47
$$

As expected, $$\begin{Vmatrix}\mathbf{y} - \mathbf{x}_3\end{Vmatrix}_2 > \begin{Vmatrix}\mathbf{y} - \hat{\mathbf{y}}\end{Vmatrix}_2$$. The solution obtained from minimizing $$RSS(\mathbf{w})$$ is truly satisfying given that a simple differentiation exercise translates to an orthogonal projection of a $$N$$-dimensional vector onto a linear subspace thereby ensuring that the residual vector is minimized. This procedure can easily be implemented on actual data and thereby providing the best linear model from which future predictions can be made on new data. 

It is often the case that the accuracy or confidence of the predictions be quantified. This is especially true when the implication of the predictions can have serious consequences if incorrect. This is where probability theory becomes useful as there are tools that allow for accuracy or confidence to be quantified. Although linear algebra provides a sound method for constructing linear models, it is evident that the notion of linear models be expanded to include a probabilistic interpretation. 

To do this, a few key assumptions must first be made. The first assumption is that each output datum $$y_i$$ is normally distributed with mean $$\mathbf{w}^T\mathbf{x}_i$$ and variance $$\sigma^2$$. This is expressed as 

$$
y_i \sim \mathcal{N}(\mathbf{w}^{T}\mathbf{x}_i, \sigma^2)
$$

 The next assumption made is that the $$N$$ samples in the vector $$\mathbf{y}$$ are independent and identically distributed (**iid**). This allows for the joint distribution of the output data to be expressed as a cumulative product of conditional probabilities. 

$$
p(\mathbf{y}|\mathbf{X}, \mathbf{\theta}) = \prod_{i}^{N}p(y_{i}|\mathbf{x}_{i}, \mathbf{\theta})
$$

where $$\mathbf{\theta}$$ are the parameters of the model. 

Consider the definition of a normal distribution with mean $$\mathbf{w}^T\mathbf{x}_i$$ and variance $$\sigma^2$$ for the *ith* sample

$$
p(y_{i}|\mathbf{x}_{i}, \mathbf{w}) = (2\pi\sigma^{2})^{-1/2}\exp(-\frac{1}{2\sigma^{2}}(y_{i}-\mathbf{w}^{T}\mathbf{x}_{i})^2)
$$

Expanding the cumulative product of normal distributions can be written as

$$
(2\pi\sigma^{2})^{-1/2}\exp(-\frac{1}{2\sigma^{2}}(y_{1}-\mathbf{w}^{T}\mathbf{x}_{1})^2)(2\pi\sigma^{2})^{-1/2}\exp(-\frac{1}{2\sigma^{2}}(y_{2}-\mathbf{w}^{T}\mathbf{x}_{2})^2) \cdots (2\pi\sigma^{2})^{-1/2}\exp(-\frac{1}{2\sigma^{2}}(y_{N}-\mathbf{w}^{T}\mathbf{x}_{N})^2)
$$

like terms can then be grouped

$$
(2\pi\sigma^{2})^{-1/2}(2\pi\sigma^{2})^{-1/2} \cdots (2\pi\sigma^{2})^{-1/2}\exp(-\frac{1}{2\sigma^{2}}(y_{1}-\mathbf{w}^{T}\mathbf{x}_{1})^2)\exp(-\frac{1}{2\sigma^{2}}(y_{2}-\mathbf{w}^{T}\mathbf{x}_{2})^2)\cdots (\exp(-\frac{1}{2\sigma^{2}}(y_{N}-\mathbf{w}^{T}\mathbf{x}_{N})^2)
$$

Using the fact that $$a^{x} \times a^{y} = a^{x+y}$$ the expression above can be simplified to

$$
\prod_{i}^{N}p(y_{i}|\mathbf{x}_{i}, \mathbf{w}) = (2\pi\sigma^{2})^{-N/2}\exp(-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_{i}-\mathbf{w}^{T}\mathbf{x}_{i})^2)
$$

This expression is the likelihood function. Notice that it contains the residual sum of squares term on the far right. The likelihood function is often expressed as the log-likelihood function as it becomes easier to work with mathematically and it also avoids numerical underflow that arises when numerous individual probabilities are multiplied. If the log of the likelihood is taken

$$
\mathcal{L}(\prod_{i}^{N}p(y_{i}|\mathbf{x}_{i}, \mathbf{w})) = \log[(2\pi\sigma^{2})^{-N/2}\exp(-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_{i}-\mathbf{w}^{T}\mathbf{x}_{i})^2)]
$$

$$
= \log[(2\pi\sigma^{2})^{-N/2}] + \log[\exp(-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_{i}-\mathbf{w}^{T}\mathbf{x}_{i})^2)]
$$

$$
= -\frac{N}{2}\log(2\pi\sigma^{2}) - \frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_{i}-\mathbf{w}^{T}\mathbf{x}_{i})^2)
$$

To estimate $$\mathbf{w}$$, it is desired to maximize the probability of the likelihood function and not minimize it. This can be interpreted as maximizing the probability of the data by selecting the optimal parameter $$\mathbf{w}$$. Although this optimization arises from maximizing probabilities, machine learning often always expresses these problems as minimization procedures. To do this, one would simply express the log-likelihood as the negative log-likelihood because maximizing the log-likelihood is equivalent to minimizing the negative log-likelihood. This post will continue to approach this optimization from maximization as it is naturally intuitive; however, it is likely that one will encounter this procedure from minimization in most literature.

To maximize the log-likelihood function, the derivative is taken with respect to $$\mathbf{w}$$. The resulting expression is set to $$0$$ and then solved for $$\mathbf{w}$$. This procedure yields the *maximum likelihood estimate* (MLE) of $$\mathbf{w}$$. The MLE is derived in matrix form as

$$
\frac{\partial}{\partial \mathbf{w}}\mathcal{L}(\prod_{i}^{N}p(y_{i}|\mathbf{x}_{i}, \mathbf{w})) = \frac{\partial}{\partial \mathbf{w}}(-\frac{N}{2}\log(2\pi\sigma^{2}) - \frac{1}{2\sigma^{2}}((\mathbf{y} - \mathbf{Xw})^{T}(\mathbf{y} - \mathbf{Xw})))
$$

$$
= \frac{\partial}{\partial \mathbf{w}}(-\frac{N}{2}\log(2\pi\sigma^{2}) - \frac{1}{2\sigma^{2}}(\mathbf{y}^{T} \mathbf{y}  - 2\mathbf{w}^{T} \mathbf{X}^T \mathbf{y} + \mathbf{w}^{T} \mathbf{X}^T \mathbf{Xw}))
$$

$$
= \frac{1}{\sigma^{2}}(\mathbf{X}^T\mathbf{y} - \mathbf{X}^{T}\mathbf{X}\mathbf{w})
$$

Note that additive terms that do not contain $$\mathbf{w}$$ will not affect the MLE. This fact will be used in further derivations. The next step is to set this expression equal to $$0$$ and solved for $$\mathbf{w}$$

$$
0 = \frac{1}{\sigma^{2}}(\mathbf{X}^T\mathbf{y} - \mathbf{X}^{T}\mathbf{X}\mathbf{w})
$$

$$
\mathbf{X}^{T}\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}
$$

$$
\hat{\mathbf{w}}_{MLE} = (\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}
$$

Thus, maximizing the likelihood function is equivalent to minimizing the residual sum of squares! This result is a bit surprising considering that $$\hat{\mathbf{w}}_{MLE}$$ is derived from the product of conditional normal distributions. Now that linear regression has been presented from a probabilistic perspective, additional methods for estimating $$\mathbf{w}$$ can be considered. The MLE of $$\mathbf{w}$$ has fundamental issues that can potentially translate to prediction errors. This is because $$\mathbf{w}$$ is chosen to best describe the training dataset $$\mathcal{D}$$; however, predictions are made on data that has not been observed. If the data are characterized by a high degree of noise, then $$\hat{\mathbf{w}}_{MLE}$$ will not generalize well and thus suffer from high prediction errors. This is commonly referred to overfitting and is a major problem associated with the MLE and by extension, the method of least squares.

A common way to limit the degree of overfitting is to constrain the weights of the linear model. This technique is known as regularization. This is done by adding a penalization term to the $$RSS(\mathbf{w})$$. The two most common methods to regularize linear models are known as ridge and lasso regression. These are

$$
\sum_{i=1}^{N}(y_{i} - w_0 - \mathbf{w^{T}x_{i}})^2 + \lambda\sum_{j=1}^{D}w_{j}^{2}
$$

and

$$
\sum_{i=1}^{N}(y_{i} - w_0 - \mathbf{w^{T}x_{i}})^2 + \lambda\sum_{j=1}^{D}|w_{j}|
$$

When the regularized linear models are presented, the inclusion of the penalty term appears to be arbitrary. The probabilistic derivations of the regularized linear models provide an intuitive interpretation of these models and thus it is important that one understands how these models are formulated. The probabilistic approach to limit overfitting is to compute the *maximum a posteriori* (MAP) estimate. MLE estimates $$\mathbf{w}$$ by maximizing the probability of the likelihood of the data given $$\mathbf{w}$$, MAP estimates $$\mathbf{w}$$ by maximizing the posterior probability of $$\mathbf{w}$$ given the data. Although this difference may appear subtle, it has significant implications on the estimate of $$\mathbf{w}$$ and on the linear regression model coefficients. This is the topic of the next section.

## Probabilistic Derivation of Regularized Linear Models

To understand the formulation of the MAP estimate, Bayes' rule will be used. Recall that Bayes' rule is 

$$
p(\theta|\mathbf{y}, \mathbf{X}) = \frac{p(\mathbf{y}|\mathbf{X}, \mathbf{\theta})p(\mathbf{\theta})}{p(\mathbf{y}|\mathbf{X})} \rightarrow posterior = \frac{likelihood * prior}{marginal\ likelihood}
$$

where the marginal likelihood is 

$$
p(\mathbf{y}|\mathbf{X}) = \int p(\mathbf{y}|\mathbf{X}, \mathbf{\theta})p(\mathbf{\theta})d\mathbf{\theta}
$$

Because the posterior requires to be maximized with respect to the parameter $$\mathbf{\theta}$$ and the marginal likelihood is independent of the parameters, only the terms in the numerator are required to be computed. Thus $$\hat{\mathbf{w}}_{MAP}$$ can be formulated as 

$$
p(\theta|\mathbf{y}, \mathbf{X}) \propto p(\mathbf{y}|\mathbf{X}, \mathbf{\theta})p(\mathbf{\theta})
$$

This formulation requires that an additional assumption be made about the distribution of the prior. The derivation of ridge regression is obtained by placing a normal distribtion on the prior. Assuming that the likelihood is normally distributed with mean $$\mathbf{w}^T\mathbf{x}$$ and variance $$\sigma^2$$ and that the prior is also normally distributed with mean $$0$$ and variance $$\tau^2$$ the posterior distribution can be written as

$$
p(\mathbf{w}|\mathbf{y}, \mathbf{X}) \propto \prod_{i}^{N}\mathcal{N}(\mathbf{w}^{T}\mathbf{x}_i, \sigma^2) \prod_{j}^{D}\mathcal{N}(0, \tau^2)
$$

$$
p(\mathbf{w}|\mathbf{y}, \mathbf{X}) \propto (2\pi\sigma^2)^{-N/2}\exp(-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^2)(2\pi\tau^2)^{-D/2}\exp(-\frac{1}{2\tau^{2}}\sum_{j=1}^{D}w_j)
$$

$$
p(\mathbf{w}|\mathbf{y}, \mathbf{X}) \propto (2\pi\sigma^2)^{-N/2}(2\pi\tau^2)^{-D/2}\exp(-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^{2}-\frac{1}{2\tau^{2}}\sum_{j=1}^{D}w_j)
$$

Taking the log of the posterior

$$
\mathcal{L}(p(\mathbf{w}\mid \mathbf{y}, \mathbf{X})) \propto \log[(2\pi\sigma^2)^{-N/2}(2\pi\tau^2)^{-D/2}\exp(-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^{2}-\frac{1}{2\tau^{2}}\sum_{j=1}^{D}w_j)]
$$

$$
= \log((2\pi\sigma^2)^{-N/2}(2\pi\tau^2)^{-D/2}) +  \log\exp(-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^{2}-\frac{1}{2\tau^{2}}\sum_{j=1}^{D}w_j)
$$

$$
= \log((2\pi\sigma^2)^{-N/2}(2\pi\tau^2)^{-D/2}) +  -\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^{2}-\frac{1}{2\tau^{2}}\sum_{j=1}^{D}w_j
$$

$$
= \log((2\pi\sigma^2)^{-N/2}(2\pi\tau^2)^{-D/2})-(\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^{2}+\frac{\sigma^{2}}{\tau^{2}}\sum_{j=1}^{D}w_j)
$$

If the leading constant term is ignored and $$\frac{\sigma^{2}}{\tau^{2}}$$ is reparameterized as $$\lambda$$, the formulation of ridge regression is recovered. This can also be expressed in matrix form as 

$$
= \log((2\pi\sigma^2)^{-N/2}(2\pi\tau^2)^{-D/2})-((\mathbf{y} - w_0\mathbf{1} - \mathbf{Xw})^{T}(\mathbf{y} - w_0\mathbf{1} - \mathbf{Xw}) + \frac{\sigma^{2}}{\tau^{2}}\mathbf{w}^T\mathbf{w})
$$

Note that in this formulation the bias term is not included in the vector $$\mathbf{x}$$. This is because the maximization of the posterior distribution should only depend on the weights of the linear model and not the bias term. Before the maximization procedure is conducted, one must consider an important data preprocessing procedure that is typical in machine learning. This procedure is to centralize  and standardize the data by subtracting the feature mean and then dividing by the feature standard deviation. Data is standardized because the scale of the input features can vary significantly and thus unstandardized data will yield different solutions for the weight vector $$\mathbf{w}$$. Additionally, the convergence rate of iterative algorithms will be accelerated if the data is standardized. Because the equation above is not in terms of the preprocessed data, it will be rewritten to reflect data preprocessing. 

The centralizing and standardization of $$\mathbf{X}$$ is written as

$$
(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}
$$

where $$\bar{\mathbf{X}}$$ is an $$N \times D$$ matrix whose columns are the average feature value of the matrix $$\mathbf{X}$$ and $$\mathbf{\Sigma}$$ is a $$D \times D$$ diagonal matrix with the feature variance along the diagonal entries of the matrix. To verify that the operation $$(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}$$ subtracts the mean from the data and then divides by the correct standard deviation value, it can be expanded as

$$
(\mathbf{X} - \bar{\mathbf{X}}) = \left[\begin{array}{c} x_{1,1} & x_{1,2} & \cdots & x_{1,D} \\ x_{2,1} & x_{2,2} & \cdots & x_{2, D} \\ \vdots & \vdots & \ddots & \vdots \\  x_{N,1} & x_{N, 2} & \cdots & x_{N, D} \end{array} \right] - \left[\begin{array}{c} \vert & \vert & \cdots & \vert \\ \bar{\mathbf{x}}_1 & \bar{\mathbf{x}}_2 & \cdots & \bar{\mathbf{x}}_D \\ \vert & \vert & \cdots & \vert \end{array} \right]
$$

$$
 = \left[\begin{array}{c} x_{1,1} - \bar{x}_1 & x_{1,2} - \bar{x}_2 & \cdots & x_{1,D} - \bar{x}_D \\ x_{2,1} - \bar{x}_1 & x_{2,2} - \bar{x}_2& \cdots & x_{2, D} - \bar{x}_D \\ \vdots & \vdots & \ddots & \vdots \\  x_{N,1} - \bar{x}_1 & x_{N, 2} -\bar{x}_2 & \cdots & x_{N, D} - \bar{x}_D\end{array} \right]
$$

$$
(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2} = \left[\begin{array}{c} x_{1,1} - \bar{x}_1 & x_{1,2} - \bar{x}_2 & \cdots & x_{1,D} - \bar{x}_D \\ x_{2,1} - \bar{x}_1 & x_{2,2} - \bar{x}_2& \cdots & x_{2, D} - \bar{x}_D \\ \vdots & \vdots & \ddots & \vdots \\  x_{N,1} - \bar{x}_1 & x_{N, 2} -\bar{x}_2 & \cdots & x_{N, D} - \bar{x}_D\end{array} \right] \left[\begin{array}{c} \sigma_1^2 & 0 & \cdots & 0 \\ 0 & \sigma_2^2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\  0 & 0 & \cdots & \sigma_{D}^2 \end{array} \right]^{-1/2} 
$$

$$
= \left[\begin{array}{c} \frac{x_{1,1} - \bar{x}_1}{\sigma_1}& \frac{x_{1,2} - \bar{x}_2}{\sigma_2} & \cdots & \frac{x_{1,D} - \bar{x}_D}{\sigma_D} \\ \frac{x_{2,1} - \bar{x}_1}{\sigma_1} & \frac{x_{2,2} - \bar{x}_2}{\sigma_2} & \cdots & \frac{x_{2, D} - \bar{x}_D}{\sigma_D} \\ \vdots & \vdots & \ddots & \vdots \\  \frac{x_{N,1} - \bar{x}_1}{\sigma_1} & \frac{x_{N, 2} -\bar{x}_2}{\sigma_2} & \cdots & \frac{x_{N, D} - \bar{x}_D}{\sigma_D}\end{array} \right]
$$

Every entry in the matrix $$(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}$$ is properly centralized by removing the corresponding feature mean and then standardized by division of the corresponding feature standard deviation and thus verifying that $$(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}$$ preprocesses the data as expected. This expression can then be substituted in for the matrix $$\mathbf{X}$$ and thus the following is obtained

$$
\log p(\mathbf{w} \mid  \mathbf{y}, \mathbf{X}) \propto \log((2\pi\sigma^2)^{-N/2}(2\pi\tau^2)^{-D/2})
$$

$$
-((\mathbf{y} - w_0\mathbf{1} - (\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w})^{T}(\mathbf{y} - w_0\mathbf{1} - (\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w}) + \frac{\sigma^{2}}{\tau^{2}}\mathbf{w}^T\mathbf{w})
$$

As mentioned in the previous section, additive terms that do not include $$\mathbf{w}$$ do not affect the maximization procedure and thus can be omitted. The following expansion of the standardized posterior maximization is extensive; however, it does provide valuable insight and is recommended to analyze and verify for ones' own understanding.

$$
-(\mathbf{y} - w_0\mathbf{1} - (\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w})^{T}(\mathbf{y} - w_0\mathbf{1} - (\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w}) - \frac{\sigma^{2}}{\tau^{2}}\mathbf{w}^T\mathbf{w}
$$

$$
= -(\mathbf{y}^T - w_0\mathbf{1}^T - \mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T})(\mathbf{y} - w_0\mathbf{1} - (\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w}) - \frac{\sigma^{2}}{\tau^{2}}\mathbf{w}^T\mathbf{w}
$$

$$
= -\mathbf{y}^{T}\mathbf{y} + \mathbf{y}^{T}(\mathbf{X} -\bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} + w_0\mathbf{y}^{T}\mathbf{1} + \mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}\mathbf{y} - \mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} 
$$

$$
- w_0\mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}\mathbf{1} + w_0\mathbf{1}^{T}\mathbf{y} - w_0\mathbf{1}^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} - w_0^{2}\mathbf{1}^{T}\mathbf{1} - \frac{\sigma^{2}}{\tau^{2}}\mathbf{w}^T\mathbf{w}
$$

The following pairs of terms are equivalent and can be combined:

$$\mathbf{y}^{T}(\mathbf{X} -\bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} = \mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}\mathbf{y}
$$ 

$$
w_0\mathbf{y}^{T}\mathbf{1} = w_0\mathbf{1}^{T}\mathbf{y}
$$

$$
w_0\mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}\mathbf{1} = w_0\mathbf{1}^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} 
$$

Notice that the last pair of terms contain an inner product between the centered data and a $$N$$-dimensional vector of ones. When data is centralized by removing the mean, the sum of these terms will equate to $$0$$ thus the following is obtained.

$$
= -\mathbf{y}^{T}\mathbf{y} + 2\mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}\mathbf{y} + 2w_0\mathbf{1}^{T}\mathbf{y} - \mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} - Nw_0^{2} - \frac{\sigma^{2}}{\tau^{2}}\mathbf{w}^T\mathbf{w}
$$

The next step is to solve for $$w_0$$. This is done by differentiating with respect to $$w_0$$ and equating the derived expression to $$0$$.

$$
\frac{\partial}{\partial{w_0}}(-\mathbf{y}^{T}\mathbf{y} + 2\mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}\mathbf{y} + 2w_0\mathbf{1}^{T}\mathbf{y} - \mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} - Nw_0^{2} - \frac{\sigma^{2}}{\tau^{2}}\mathbf{w}^T\mathbf{w})
$$

$$
= 2(\mathbf{1}^{T}\mathbf{y}) - 2Nw_{0}
$$

setting this expression equal to $$0$$

$$
0 = 2(\mathbf{1}^{T}\mathbf{y}) - 2Nw_{0}
$$

$$
Nw_{0} = \mathbf{1}^{T}\mathbf{y}
$$

$$
w_{0} = \frac{1}{N}\mathbf{1}^{T}\mathbf{y} \rightarrow \frac{1}{N}\sum_{i=1}^{N}y_i = \bar{y}
$$

Thus, arriving at the conclusion that $$w_{0}$$ is the average of the output data $$\mathbf{y}$$ when the data is standardized. Notice that after differentiating with respect to $$w_{0}$$ the expression does not depend on the matrix $$\mathbf{\Sigma^{-1/2}}$$ this implies that the same result is obtained for $$w_{0}$$ if the data is centered without standardizing. This makes sense given that $$\mathbf{\Sigma^{-1/2}}$$ only rescales the data and does not shift it and thus it does not affect the location of optimal values. Lastly, the optimal values of the weight vector $$\mathbf{w}$$ can be computed.

$$
\frac{\partial}{\partial{\mathbf{w}}}(-\mathbf{y}^{T}\mathbf{y} + 2\mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}\mathbf{y} + 2w_0\mathbf{1}^{T}\mathbf{y} - \mathbf{w}^{T}\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} - Nw_0^{2} - \frac{\sigma^{2}}{\tau^{2}}\mathbf{w}^T\mathbf{w})
$$

$$
= 2\mathbf{\Sigma^{-1/2}}(\mathbf{X} - \bar{\mathbf{X}})^{T}\mathbf{y} - 2\mathbf{\Sigma^{-1/2}}(\mathbf{X} - \bar{\mathbf{X}})^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} - 2\frac{\sigma^{2}}{\tau^{2}}\mathbf{w}
$$

Setting equal to $$0$$ and then solving for $$\mathbf{w}$$

$$
\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}\mathbf{w} + \frac{\sigma^{2}}{\tau^{2}}\mathbf{w} = \mathbf{\Sigma^{-1/2}}(\mathbf{X}- \bar{\mathbf{X}})^{T}\mathbf{y}
$$

$$
(\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2} + \frac{\sigma^{2}}{\tau^{2}}\mathbf{I})\mathbf{w} = \mathbf{\Sigma}^{-1/2}(\mathbf{X}- \bar{\mathbf{X}})^{T}\mathbf{y}
$$

$$
\hat{\mathbf{w}}_{s-Ridge} = (\mathbf{\Sigma}^{-1/2}(\mathbf{X} - \bar{\mathbf{X}})^{T}(\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2} + \frac{\sigma^{2}}{\tau^{2}}\mathbf{I})^{-1}\mathbf{\Sigma}^{-1/2}(\mathbf{X}- \bar{\mathbf{X}})^{T}\mathbf{y}
$$

for clarity, one can define the standardized data matrix as

$$
\mathbf{X}_{s} = (\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2}
$$

The standardized weights can then be defined as 

$$
\hat{\mathbf{w}}_{s-Ridge} = (\mathbf{X}_{s}^{T}\mathbf{X}_{s} + \frac{\sigma^{2}}{\tau^{2}}\mathbf{I})^{-1}\mathbf{X}_{s}^{T}\mathbf{y}
$$

If the data is not standardized but it is only centralized, the solution  will be of the same form. The centralized data matrix is defined as

$$
\mathbf{X}_c = (\mathbf{X} - \bar{\mathbf{X}})
$$

and the solution using centralized data is

$$
\hat{\mathbf{w}}_{c-Ridge} = (\mathbf{X}_{c}^{T}\mathbf{X}_{c} + \frac{\sigma^{2}}{\tau^{2}}\mathbf{I})^{-1}\mathbf{X}_{c}^{T}\mathbf{y}
$$

The standardized solution can also be written in terms of the centralized matrix $$\mathbf{X}_c$$ (this is true only if the data is both centralized and standardized).

$$
\mathbf{X}_s = (\mathbf{X} - \bar{\mathbf{X}})\mathbf{\Sigma}^{-1/2} = \mathbf{X}_{c}\mathbf{\Sigma}^{-1/2} \mbox{ and } \mathbf{X}_{s}^{T} = \mathbf{\Sigma}^{-1/2}\mathbf{X}_{c}^{T}
$$

$$
\hat{\mathbf{w}}_{s-Ridge} = (\mathbf{X}_{s}^{T}\mathbf{X}_{s} + \frac{\sigma^{2}}{\tau^{2}}\mathbf{I})^{-1}\mathbf{X}_{s}^{T}\mathbf{y} 
$$

$$
= (\mathbf{\Sigma}^{-1/2}\mathbf{X}_{c}^{T}\mathbf{X}_{c}\mathbf{\Sigma}^{-1/2} + \frac{\sigma^{2}}{\tau^{2}}\mathbf{I})^{-1}\mathbf{\Sigma}^{-1/2}\mathbf{X}_{c}^{T}\mathbf{y}
$$

Notice that the expression above is identical to the *least-squares* solution with the exception of the diagonal matrix $$\frac{\sigma^{2}}{\tau^{2}}\mathbf{I}$$. A natural question at this point is how does the inclusion of the diagonal matrix $$\frac{\sigma^{2}}{\tau^{2}}\mathbf{I}$$ address the overfitting issue that occurs when the likelihood is maximized. This can be explained by analyzing the assumption that was made about the distribution of the parameters $$\mathbf{w}$$. 

Recall that the prior was assumed to be normally distributed with $$0$$ mean and variance $$\mathbf{\tau}^{2}$$. This assumption implies that $$\mathbf{w}$$ is likely to be $$0$$ with some distribution $$\mathbf{\tau}^{2}$$. Thus, the inclusion of the prior encourages the terms in the vector $$\mathbf{w}$$ to be localized about $$0$$. The level at which it is encouraged that the parameters $$\mathbf{w}$$ approach zero is controlled by the variance $$\mathbf{\tau}^{2}$$. The figure below is a visualization of this idea. Notice that as $$\mathbf{\tau}^{2}$$ increases, the mass of the density becomes less concentrated around $$0$$. This can be interpreted as one being less confident that the terms in the vector $$\mathbf{w}$$ are close to $$0$$. If these  terms should be close to zero, the variance $$\mathbf{\tau}^{2}$$ will be small and reflects one's certainty of this claim. The parameter $$\frac{\sigma^{2}}{\tau^{2}}$$ is usually presented as $$\lambda$$ in ridge regression. The figure on the right displays the relationship between $$\tau^{2} \mbox{ and } \lambda$$. As $$\tau^{2}$$ increases, $$\lambda$$ decreases, thus a small value of $$\lambda$$ corresponds to a normal distribution with a large standard deviation. This can be interpreted as less confidence that the values in the weight vector $$\mathbf{w}$$ are $$0$$.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/GaussianPrior.html" height="525" width="100%"></iframe>

The effects of regularization using a normal distribution for the prior can be demonstrated with the following example. A toy dataset was generated using $$y=2.5x+3.28$$ and adding white noise. A total of 200 samples were used thus the matrix $$\mathbf{X}$$ and the corresponding output vector $$\mathbf{y}$$ are of size $$200 \times 1$$. The standardized weight values are computed at increasing values of lambda. 

```python
I = np.identity(X.shape[1])
lambdas_ = [0, 1, 10, 100, 1000, 10000]
for lambda_ in lambdas_:
    print(np.around(np.linalg.inv((lambda_ * I) + X_s.T @ X_s) @ X_s.T @ \
    y, decimals=3)[0])

6.99
6.956
6.657
4.66
1.165
0.137
```
These results agree with the previous assertions, as lambda increases (decreasing $$\tau^2$$) the calculated weights are encouraged to be $$0$$ and thus the weight of the model approaches $$0$$. The results can be visualized in the figure below. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/RegLinRegress.html" height="525" width="100%"></iframe>

Notice that as $$\lambda$$ increases, the slope of the line decreases. Model regularization decreases the ability of the MLE to adhere to the training data and thus may improve the ability of the model to generalize on unobserved data. Recall that ridge regression was derived by placing a normal distribution on prior. If it is believed that this assumption is incorrect, another distribution can be placed. Another common distribution placed on the prior is a laplace distribution. Although the results lead to another regularized model, the laplace distribution modifies certain aspects of the model's behavior which differ significantly from ridge regression. To demonstrate the differences between placing a normal and laplace distribution on the prior, the regularized model with a laplace prior will be derived. To begin, a laplace distribution with  $$0$$ mean and a scaling parameter $$b =  \frac{1}{\lambda}$$ will be assumed. This is formulated as

$$
p(\mathbf{w}|\mathbf{y}, \mathbf{X}) \propto \prod_{i}^{N}\mathcal{N}(\mathbf{w}^{T}\mathbf{x}_i, \sigma^2) \prod_{j}^{D}Lap(0, \frac{1}{\lambda})
$$

$$
= (2\pi\sigma^2)^{-N/2}\exp(-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^2)(\frac{\lambda}{2})^{D}\exp(-\lambda\sum_{j=1}^{D}|w_j|)
$$

taking the log and omitting the constant yields

$$
= (-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^2) + (-\lambda\sum_{j=1}^{D}|w_j|)
$$

$$
= -(\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^2 + 2\sigma^{2}\lambda\sum_{j=1}^{D}|w_j|)
$$

Reparameterizing $$2\sigma^{2}\lambda \mbox{ as } \lambda^{'}$$, it can be seen that the formulation for the lasso regression model is recovered. The next step is to maximize the posterior probability; however, the absolute value term that occurs from the laplace distribution complicates this procedure. The maximization procedure that has been carried out thus far is to differentiate the function and obtain an analytical solution for the weight vector $$\mathbf{w}$$. This is no longer an option because the absolute value is not differentiable. To estimate the weight vector $$\mathbf{w}$$ the procedure that has been used for the previous solutions must be modified. The modified procedure will be an iterative method that will solve for each weight individually until a prespecified error tolerance is reached. To demonstrate this procedure, the expression above must be redefined in terms of j-th component. 

$$
-(\sum_{i=1}^{N}(y_i-w_0-\mathbf{w}^{T}\mathbf{x}_i)^2 + 2\sigma^{2}\lambda\sum_{j=1}^{D}|w_j|) = -(\sum_{i=1}^{N}(y_i - w_0 - x_{i,j}w_j - \mathbf{w}_{-j}^{T}\mathbf{x}_{i,-j})^2 + 2\sigma^{2}\lambda\sum_{j=1}^{D}|w_j|)
$$

The terms with the subscript $$_{-j}$$ denote a vector that does not include the jth component. For example, if $$D=4$$, the weight vector $$\mathbf{w}^T$$ is

$$
\mathbf{w}^{T} = \left[\begin{array}{c} w_1 & w_2 & w_3 & w_4 \end{array} \right] 
$$

thus, when $$j=2$$, the weight vector $$\mathbf{w}_{-j}^{T}$$ is

$$
\mathbf{w}_{-j}^{T} = \left[\begin{array}{c} w_1 & w_3 & w_4 \end{array} \right] 
$$

The modified expression will then allow for differentiation with respect to each component individually. 

$$
\frac{\partial}{\partial{w_j}}(-(\sum_{i=1}^{N}(y_i - w_0 - x_{i,j}w_j - \mathbf{w}_{-j}^{T}\mathbf{x}_{i,-j})^2 + 2\sigma^{2}\lambda\sum_{j=1}^{D}|w_j|))
$$

$$
= (-(-2x_{i,j}\sum_{i=1}^{N}(y_i - w_0 - x_{i,j}w_j - \mathbf{w}_{-j}^{T}\mathbf{x}_{i,-j})) +  2\sigma^{2}\lambda\frac{\partial}{\partial{w_j}}|w_j|)
$$

The terms can be grouped as follows

$$
= -(2w_{j}\sum_{i=1}^{N}x_{i,j}^{2} -2\sum_{i=1}^{N}x_{i,j}(y_i - w_0 - \mathbf{w}_{-j}^{T}\mathbf{x}_{i,-j}) + 2\sigma^{2}\lambda\frac{\partial}{\partial{w_j}}|w_j|)
$$

This expression can be simplified as

$$
= -(w_{j}a_{j} - c_{j} + 2\sigma^{2}\lambda\frac{\partial}{\partial{w_j}}|w_j|)
$$

where

$$
a_{j} = 2\sum_{i=1}^{N}x_{i,j}^{2} 
$$

$$
c_{j} = 2\sum_{i=1}^{N}x_{i,j}(y_i - w_0 - \mathbf{w}_{-j}^{T}\mathbf{x}_{i,-j})
$$

Focus can now be placed on the derivative of the absolute value term

$$
\frac{\partial}{\partial{w_j}}|w_j|
$$

To differentiate this term, the notion of a derivative is extended. A subderivative of a convex function $$f : \mathcal{I} \rightarrow  \mathbb{R}$$ at a point $$x_0$$ is defined as a scalar $$c$$ such that

$$
f(x) - f(x_0) \geq c(x-x_0) \forall x \in  \mathcal{I} \space \mbox{and} x_0 \in \mathcal{I}
$$

This means that the difference of $$c$$ at $$x_0$$ and $$x$$ is at most equal to $$f(x) - f(x_0)$$ in the interval $$\mathcal{I}$$. When this definition holds, a *set* of subderivatives is defined as the interval $$[a, b]$$ where $$a$$ and $$b$$ are defined as 

$$
a = \lim_{x\rightarrow x_{0}^{-}} \frac{f(x)-f(x_0)}{x-x_0}, b = \lim_{x\rightarrow x_{0}^{+}} \frac{f(x)-f(x_0)}{x-x_0}
$$

This set of subderivatives is known as the subdifferential of $$f$$ at $$x_0$$ and is denoted as $$\partial f(x) \vert _{x_0}$$. When the subderivative is applied to the absolute value $$\vert w_j \vert$$ the following result is obtained

$$
\partial f(|w_j|) = \left\{ \begin{array}{rcl} -1 & \mbox{if} & w_j < 0 \\ [-1, 1] & \mbox{if} & w_j = 0\\1 & \mbox{if} & w_j > 0 \end{array}\right.
$$

Substituting these results into $$-(w_{j}a_{j} - c_{j} + 2\sigma^{2}\lambda\frac{\partial}{\partial{w_j}}\vert w_j\vert)$$ yields

$$
\partial f(|w_j|) = \left\{ \begin{array}{rcl} -(w_{j}a_{j} - c_{j} - \lambda^{'}) & \mbox{if} & w_j < 0 \\ [- c_{j} - \lambda^{'}, - c_{j} + \lambda^{'}] & \mbox{if} & w_j = 0\\ -(w_{j}a_{j} - c_{j} + \lambda^{'}) & \mbox{if} & w_j > 0 \end{array}\right.
$$

for clarity, the term $$2\sigma^{2}\lambda$$ is reparameterized as $$\lambda^{'}$$. Setting each expression equal to $$0$$ and solving for $$w_j$$

$$
\partial f(|w_j|) = \left\{ \begin{array}{rcl} w_{j} = \frac{c_{j} + \lambda^{'}}{a_{j}} & \mbox{if} & c_j < -\lambda^{'} \\ w_j = 0 & \mbox{if} & c_j \in [-\lambda^{'}, \lambda^{'}]\\ w_{j} = \frac{c_{j} - \lambda^{'}}{a_{j}}& \mbox{if} & c_j > \lambda^{'}  \end{array}\right.
$$

Note that the $$w_j$$ depends on the value of $$c_j$$. This is because $$\lambda^{'} \mbox{and } a_j \mbox{ are} \geq 0$$ as $$\lambda^{'}$$ captures the deviation of the gaussian and laplace distributions and $$a_j$$ is a sum of squared values, both of which cannot be negative. Thus, in the first solution of $$w_j$$, $$c_j$$ must be less than the negative value of $$\lambda^{'}$$ for $$w_j$$ to be less than $$0$$. The same reasoning can be applied for the other solutions of $$w_j$$. To implement these results on a dataset, the following algorithm can be followed:

1) Initialize $$\mathbf{w} = (\mathbf{X}^{T}\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^{T}\mathbf{y}$$ or any random value

2) while: *not converged*

3) for j = 1, $$\cdots$$,D:

4) $$a_{j} = 2\sum_{i=1}^{N}x_{i,j}^{2}$$

5) $$c_{j} = 2\sum_{i=1}^{N}x_{i,j}(y_i - w_0 - \mathbf{w}_{-j}^{T}\mathbf{x}_{i,-j})$$

6) $$\left\{ \begin{array}{rcl} w_{j} = \frac{c_{j} + \lambda^{'}}{a_{j}} & \mbox{if} & c_j < -\lambda^{'} \\ w_j = 0 & \mbox{if} & c_j \in [-\lambda^{'}, \lambda^{'}]\\ w_{j} = \frac{c_{j} - \lambda^{'}}{a_{j}}& \mbox{if} & c_j > \lambda^{'}  \end{array}\right.$$


The implementation of this algorithm in python is simple. The code below carries out the pseudocode above.

```python
#Specifiy lambda and error tolerance
lambda_ = 10
tol = 1e-4
#Initialize weight vector as ridge solution
w = np.linalg.inv((lambda_ * I) + X_s.T @ X_s) @ X_s.T @ y
check = True
#While loop terminates when check variable is False
while check:
    #create a copy of weight vector to compare at the end of for loop
    w_prev = np.copy(w)
    for j in range(X_s.shape[1]):
        mask = np.ones(X_s.shape[1], dtype=bool)
        mask[j] = False
        aj = 2 * (X_s[:, ~mask].T @ X_s[:, ~mask])
        cj = 2 * (X_s[:, ~mask].T @ (y - X_s[:, mask] @ w[mask]))
        if cj >= -lambda_ and cj <= lambda_:
            wj = 0
        elif cj < -lambda_:
            wj = (cj + lambda_) / aj
        elif cj > lambda_:
            wj = (cj - lambda_) / aj
        w[~mask] = wj
    error = np.sum(abs(w_prev - w))
#At the end of for loop, check if current error is below the tolerance
    if error < tol:
        check = False
```

To compare the differences between placing a normal and laplace distribution prior on the weight vector, these distributions must be analyzed and compared. The figure below displays a normal and laplace distribution. Note that the peak of the laplace distribution about its mean is higher than that of the gaussian. This implies that the laplace distribution places a larger probability that the components of the weight vector are $$0$$. This distinction from the gaussian prior gives the sparsity-promoting property to this version of regularized linear regression. To demonstrate why one may select lasso regression instead or ridge regression, the models will be applied on a real dataset in the following section.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/Laplace_Gaussian PDFS.html" height="525" width="100%"></iframe>

## Linear Regression on Prostate Dataset

The dataset that will be used in this example is the prostate dataset which is located at *https://web.stanford.edu/~hastie/ElemStatLearn//datasets/prostate.data*. This data was used in a study by Stamey et al. (1989) which examined the correlation between the level of prostate specific antigen (PSA) and other clinical measures in $$97$$ men. The goal was to predict the log of PSA (*lpsa*) from the set of inputs that included log cancer volume (*lcavol*), log prostate weight (*lweight*), age, log of benign prostatic hyperplasia amount (*lbph*), seminal vesicle invasion (*svi*), log of capsular penetration (*lcp*), Gleason score (*gleason*), and percent of Gleason scores $$4$$ and $$5$$ (*pgg45*).The data is first loaded as a pandas dataframe and the first four rows are displayed for inspection. 

```python
df = pd.read_table("https://web.stanford.edu/~hastie/ElemStatLearn//datasets/prostate.data")

df.head()
```

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/Prostate Dataset.html" height="200" width="100%"></iframe>


The column titled *Unnamed: 0* is an index column and thus will be dropped. Also, note the last column titled *train*. This column differentiates the test set from the training set via the label values "T" and "F". The value "T" indicates ```True``` and thus the corresponding data row belongs to the training data. This data will form $$\mathcal{D} = \{(\mathbf{x}_i , y_i)\}^N_{i=1}$$. The index column is dropped and the data is parsed into a training dataset.

```python
df_train = df[df["train"] == 'T'] #Training Data
df_train = df_train.iloc[:,1:] #Remove unammed column
df_train = df_train.iloc[:, :-1] #Remove train column
```

The training dataset $$\mathcal{D}$$ is then inspected to determine what data type is contained within the features.

```python   
df_train.info()

<class pandas.core.frame.DataFrame>
Int64Index: 67 entries, 0 to 95
Data columns (total 9 columns):
 #   Column   Non-Null Count  Dtype  
"---  ------   --------------  -----"  
 0   lcavol   67 non-null     float64
 1   lweight  67 non-null     float64
 2   age      67 non-null     int64  
 3   lbph     67 non-null     float64
 4   svi      67 non-null     int64  
 5   lcp      67 non-null     float64
 6   gleason  67 non-null     int64  
 7   pgg45    67 non-null     int64  
 8   lpsa     67 non-null     float64
dtypes: float64(5), int64(4)
memory usage: 7.7 KB
```

This output reveals that the features *lcavol, lweight, lbph, lcp,* and *lpsa* are floating-point numbers, and the remaining features are integers. The data is further inspected by plotting the histograms of each feature.


<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/ProstateData_Histograms.html" height="525" width="100%"></iframe>

Note that *svi* and *gleason* are categorical values. Another useful plot that allows for data inspection is a scatter plot matrix. The plot below displays the correlation amongst the features and the output. 

<div class="scroll_box">
<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/Prostate Dataset SPLOM.html" height="900" width="900"></iframe>
</div>

Lastly, a heatmap of the correlation coefficients is generated and analyzed. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/Corr_Heatmap.html" height="525" width="100%"></iframe>

The features and the output vector are separated and turned into numpy arrays and the data matrix is standardized

```python
X = df_train.iloc[:, :-1].to_numpy()
y = df_train.iloc[:, -1:].to_numpy()
X = (X - np.mean(X, axis=0)) / np.sqrt(np.var(X, axis=0))
```

For ordinary least squares, a vector of ones is prepended onto the data matrix $$\mathbf{X}$$ and then the weight vector is computed.

```python
X_ = np.c_[np.ones(X.shape[0]), X]
w_ls = np.linalg.inv(X_.T @ X_) @ X_.T @ y
print(np.around(w_ls, decimals=3))

[[ 2.452]
 [ 0.711]
 [ 0.29 ]
 [-0.141]
 [ 0.21 ]
 [ 0.307]
 [-0.287]
 [-0.021]
 [ 0.275]]
```

Before least squares is compared to the ridge and lasso solution, lets first observe the effects of these methods on the estimated weight vectors. The analytical ridge solution and the lasso iterative solutions are calculated at varying values of $$\lambda$$ using the methods that were demonstrated above. The figure below displays how the weight vectors change.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/l2_and_l1 coefficients.html" height="525" width="100%"></iframe>

Although the weights in both solutions approach zero as the degree of regularization is increased, the paths of the weight vectors differ significantly. The sparsity property of the lasso solution is evident. After a coefficient equates to $$0$$, in no longer becomes active. This is not true for the ridge solution. While a weight may be equal to $$0$$ at a certain degree of regularization, a small change in regularization will activate those terms again. For example, consider the *lcp* feature in the ridge solution. The coefficient value is initially zero. As regularization is increased, the value approaches $$0$$ and then becomes positive. Thus, *lcp* must be $$0$$ at one point in the regularization path but it does not remain at $$0$$. Because certain input features may not help predictions or may be uncorrelated with the data, lasso regression may be implemented to determine which features are omitted from the final model. This would reveal valuable information as it could suggest that certain data can be omitted from the data gathering phase. Now that these differences are understood, at which value should $$\lambda$$ be set to so that the final models can be compared? This is usually done via the process of *cross-validation*.

In machine learning data is usually separated into a training set and test set. The testing set is never observed until a final model is selected. For model selection, the training set is further divided into another subset called a validation set. The purpose of the validation set is to adjust model hyperparameters without biasing results towards the test dataset and avoid overfitting to the training dataset. This is demonstrated in the figure below. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/Lambda_Validation.html" height="525" width="100%"></iframe>

Both plots contain the training and validation root mean squared error (RMSE) as a function of lambda. Notice that when the training error is minimized, the validation error has yet to reach its minimum. The validation error gives insight into how well the current model can generalize because the model was not exposed to this data to estimate the weights of the model and thus gives a better indication of what the test error will be. The lambda that was chosen for the ridge and lasso solutions corresponds to the validation error minimum. The estimated weights and final test error are displayed in the table below.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/LinearRegression/Test_Error.html" height="250" width="100%"></iframe>

Note that the model with the lowest test error was achieved by lasso regression. This does not suggest that lasso regression is better than ridge or least squares, it just happened to produce better results in this modeling process. Factors that could yield different results can be attributed to the size of the cross validation set and the way it was generated. Recall that the histograms display differing proportions in the sample distribution of each feature. If the cross validation dataset does not reflect these distributions accurately, it will impact the cross validation error curve such that it may favor one model over another. 

The size of the training set may also contribute to these results because smaller datasets are more prone to the effects of the variance. This training dataset was made from a total of $$67$$ samples. For the cross validation set, a total of $$20$$ samples were randomly selected. Depending on the variance of the data, $$20$$ samples may not reflect the overall variance accurately and thus skew the cross validated error curve. These effects should always be considered when generating models. Because this is not the topic of this post, methods for addressing these issues will are not explored.

The last point that should be noted is that the code that was included for the least squares and ridge regression solutions is not what one will typically encounter in practice. Matrix inversion is computationally costly and numerically unstable. Prior to inverting, the matrix $$\mathbf{X}$$ is decomposed via the *QR* decomposition. This process turns $$\mathbf{X}$$ into an orthogonal matrix $$\mathbf{Q}$$ and an upper triangular matrix $$\mathbf{R}$$ which are far easier to invert and thus accelerate the process of matrix inversion. The code was demonstrated in this manner as a method to establish the link between the derived mathematical solutions and their implementation in python as it helps those who may be unfamiliar with translating math into working python code. The use of this code in this example was applicable because the small size of the dataset; however, for larger datasets, the long computation times will render this code impractical and thus standard optimized code for regularized linear regression should be used. 

To conclude, it was observed that the method of least squares provides the best-fit such that the residual sum of squares is minimized in the training data. This procedure can also be derived from a probabilistic perspective by assuming that the output data is normally distributed about its mean $$\mathbf{w}^{T}\mathbf{x}$$ with variance $$\sigma^{2}$$ and yields the MLE of $$\mathbf{w}$$. Given the MLE can suffer from overfitting, regularized linear models were then explored. These models can be derived by computing the MAP estimate of $$\mathbf{w}$$ which is done by inclusion of the prior. By placing a normal or laplace distribution on the prior, the ridge and lasso regression models were recovered. Both models encourage the weights to be $$0$$; however, lasso regression possesses a sparsity promoting property that the ridge does not. This was observed by analyzing the ridge and lasso paths of the weight vector on the prostate dataset as regularization of the model was increased. These demonstrations display how introductory machine learning models can be derived by incorporating a probabilistic perspective. Adopting this perspective is advantageous for those who are beginning to learn topics in machine learning as it will facilitate comprehension of more complex topics and allow for a better way to work with the inherent uncertainty in the data and the model-generating process.

