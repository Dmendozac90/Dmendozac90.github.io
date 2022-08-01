---
layout: post
title:  "Bayesian Linear Regression: Conjugate Analysis and MCMC with Tensorflow"
date:   2022-02-19 -
description: "Bayesian linear regression differs from point estimates in that the posterior distribution over the parameters and the predictive distribution are computed. This provides an additional component of information because the uncertainty associated with the parameters and estimates made from the model are well-defined. To obtain an analytical solution of the posterior distributions, one is restricted to selecting a class of priors that are known as conjugate priors. This results in a posterior distribution of the same form of the prior and thus the distribution parameters can be computed. When non-conjugate priors are desired to model parameters, the ability to analytically derive the parameters of the posterior is no longer possible. This forces one to be restricted to conjugate priors or explore other methods like Markov chain Monte Carlo to estimate posterior distributions. This post displays how to compute the posterior distribution using a conjugate prior over the model parameters in the scenario in which the variance of a synthetic dataset is assumed to be known. Additionally, the Hamiltonian Markov Chain algorithm will also be implemented using the Tensorflow Probability library to replicate the same results from the conjugate analysis to become familiar with Markov chain Monte Carlo methods and how they may be extended when a conjugate prior is not selected."
categories: Tensorflow Bayesian_Linear_Regression Markov_Chain_Monte_Carlo Conjugate_Analysis
html_url: /assets/img/BayLinReg/Predictive_posterior_4.webp
---

**Outline**
-   [Introduction](#introduction)
-   [Posterior Derivation](#posterior-derivation)
-   [Bayesian Linear Regression Application](#bayesian-linear-regression-application)
-   [Hamiltonian Monte Carlo](#HMCMC)

## Introduction
Bayesian linear regression has a significant advantage when compared to the maximum likelihood (MLE) and the maximum a posteriori (MAP) estimates in that the associated uncertainty of the model parameters and model predictions are computed. Uncertainty estimates are important because it provides additional insight when making decisions based on model results. Consider a scenario in which an investment decision will be made on estimates provided by a model that demonstrate significant positive returns for an investor. Without any knowledge of the associated uncertainty, this decision may lead to significant losses and thus it is highly beneficial to quantify associated model uncertainty. If instead, optimistic results were provided by this model but with a large degree of uncertainty, one may not be willing to incur this risk and opt to pursue a different investment. Similar scenarios occur in other risk-adverse settings where understanding associated uncertainty is beneficial to guide actions in an informed manner.

Linear regression models are a great setting to begin exploring Bayesian inference topics and obtain intuition because of the computational simplicity and the ability to visualize associated distributions. Its also easy to extend results to begin to understand more advanced modeling methods like Markov chain Monte Carlo (MCMC) algorithms. To build intuition for Bayesian linear regression, the scenario in which the noise variance is known will be assumed. The likelihood and prior will then be defined and the posterior will be derived. Deriving the posterior distribution requires a brief digression of the formulation of a multivariate standard quadratic and its corresponding form when the square is completed. Although the mathematics of this derivation is lengthy, it is recommended that one becomes familiar with this derivation as it provides valuable insight to any practitioner implementing Bayesian methods for inference. 

After the posterior and predictive posterior distributions are established, Bayesian linear regression will be sequentially implemented on a synthetic dataset in which the underlying model parameters will be known. This approach will demonstrate the intuitive nature of model variance and convergence as observed data is incrementally observed. The last section of this post will then reconstruct the analytical results using Tensorflow Probability to build a Hamiltonian Markov Chain (HMC). This will be a brief section as its purpose is only to demonstrate a simple example of how to begin using this great resource from probabilistic models. 


## Posterior Derivation

To compute the posterior distribution over the parameters, Bayes' rule will be applied by specifying a likelihood and prior distribution. Recall that Bayes' rule is defined as

$$
p(\theta|\mathbf{y}, \mathbf{X}) = \frac{p(\mathbf{y}|\mathbf{X}, \mathbf{\theta})p(\mathbf{\theta})}{p(\mathbf{y}|\mathbf{X})} \rightarrow posterior = \frac{likelihood * prior}{marginal\ likelihood}
$$

For simplicity, the marginal likelihood will not be considered and thus Bayes' rule will be considered as a proportion

$$
p(\theta|\mathbf{y}, \mathbf{X}) \propto p(\mathbf{y}|\mathbf{X}, \mathbf{\theta})p(\mathbf{\theta})\rightarrow posterior \propto likelihood * prior
$$

In linear regression, the likelihood is given by 

$$
p(\mathbf{y | X, w}, \mu{}, \sigma{}^2) = \mathcal{N}(\mathbf{y} | \mu{} + \mathbf{Xw}, \sigma{}^{2}\mathbf{I}_{N}) \propto \exp{(-(2\sigma{}^{2})^{-1}(\mathbf{y} - \mu\mathbf{1}_{N}-\mathbf{Xw})^{T}(\mathbf{y} - \mu\mathbf{1}_{N}-\mathbf{Xw}))}
$$

To obtain an analytical solution to the posterior, a conjugate prior is selected. Doing so will yield a posterior of the same form as the prior. The conjugate prior to the Gaussian likelihood is also Gaussian and is denoted by

$$
p(\mathbf{w}) = \mathcal{N}(\mathbf{w}|\mathbf{w}_{0}, \mathbf{V}_{0})
$$

This yields the following posterior 

$$
p(\mathbf{w}|\mathbf{X, y}, \sigma{}^{2}) \propto \mathcal{N}(\mathbf{w}|\mathbf{w}_{0}, \mathbf{V}_{0})\mathcal{N}(\mathbf{y | X, w}, \sigma{}^{2}\mathbf{I}_{N}) = \mathcal{N}(\mathbf{w}|\mathbf{w}_{N}, \mathbf{V}_{N})
$$

Before the Gaussian posterior is derived, a quick overview of multivariate quadratic forms will be presented. This is required to express the product of the prior and likelihood in a form that will allow for the posterior mean and variance to be computed. A multivariate quadratic is written in standard form as follows

$$
\mathbf{x}^{T}\mathbf{Ax} + \mathbf{x}^{T}\mathbf{b} + c
$$

A standard quadratic can be reformulated as follows by completing the square

$$
(\mathbf{x - d})^{T}\mathbf{A}(\mathbf{x - d}) + e 
$$

This expression can be expanded into 

$$
\mathbf{x}^{T}\mathbf{Ax} - 2\mathbf{x}^{T}\mathbf{Ad} + \mathbf{d}^{T}\mathbf{Ad} + e
$$

Recall that a multivariate Gaussian distribution is 

$$
\mathcal{N}(\mathbf{x}|\mathbf{\mu, \Sigma}) \propto \exp((-2)^{-1}(\mathbf{x - \mu})^{T}\mathbf{\Sigma}^{-1}(\mathbf{x - \mu}))
$$

Notice that the expression within the exponential resembles a quadratic expression. It can be seen that 

$$
\mathbf{d} = \mathbf{\mu}
$$

and 

$$
\mathbf{A} = \mathbf{\Sigma}^{-1}
$$

Returning to the posterior distribution, the multiplication of the likelihood and prior is expanded as follows

$$
\mathcal{N}(\mathbf{y | X, w},\mathbf{V}_{0}) \sigma{}^{2}\mathbf{I}_{N})\mathcal{N}(\mathbf{w}|\mathbf{w}_{0}) = 
$$

$$
\exp{(-(2^{-1})((\mathbf{y}-\mu\mathbf{1}_{N}-\mathbf{Xw})^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}(\mathbf{y}-\mu\mathbf{1}_{N}-\mathbf{Xw})))}\exp{(-(2^{-1})((\mathbf{w}-\mathbf{w}_{0})^{T}\mathbf{V}_{0}^{-1}(\mathbf{w}-\mathbf{w}_{0})))}
$$

$$
 = \exp{(-(2^{-1})((\mathbf{y}-\mu\mathbf{1}_{N}-\mathbf{Xw})^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}(\mathbf{y}-\mu\mathbf{1}_{N}-\mathbf{Xw})+(\mathbf{w}-\mathbf{w}_{0})^{T}\mathbf{V}_{0}^{-1}(\mathbf{w}-\mathbf{w}_{0})))}
$$

To keep track of all the terms when expanding the expression above, they will be evaluated sequentially and then like-terms will be combined. The following terms are obtained

$$
\mathbf{y}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{y} -\mathbf{y}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mu\mathbf{1}_N-\mathbf{y}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{Xw}
$$

$$
-\mu\mathbf{1}^{T}_{N}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{y} + \mu\mathbf{1}^{T}_{N}(\sigma^{2}\mathbf{I}_{N})^{-1}\mu\mathbf{1}_{N}+\mu\mathbf{1}^{T}_{N}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{Xw}
$$

$$
-\mathbf{w}^{T}\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{y}+\mathbf{w}^{T}\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mu\mathbf{1}_{N}+\mathbf{w}^{T}\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{Xw}
$$

$$
\mathbf{w}^{T}\mathbf{V}_{0}^{-1}\mathbf{w}-\mathbf{w}^{T}\mathbf{V}_{0}^{-1}\mathbf{w}_{0}-\mathbf{w}_{0}^{T}\mathbf{V}_{0}^{-1}\mathbf{w}+\mathbf{w}_{0}^{T}\mathbf{V}_{0}^{-1}\mathbf{w}_{0}
$$

The posterior distribution only depends on the parameters $$\mathbf{w}$$. The terms are grouped according to the quadratic and linear interactions with $$\mathbf{w}$$ and those that are independent of $$\mathbf{w}$$.

The quadratic terms are grouped initially

$$
\mathbf{w}^{T}\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{Xw} + \mathbf{w}^{T}\mathbf{V}_{0}^{-1}\mathbf{w} = \mathbf{w}^{T}(\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{X} + \mathbf{V}_{0}^{-1})\mathbf{w}
$$

Next, the linear terms are grouped

$$
-2\mathbf{w}^{T}\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{y}+2\mathbf{w}^{T}\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mu\mathbf{1}_{N}-2\mathbf{w}^{T}\mathbf{V}_{0}^{-1}\mathbf{w}_{0} = -2\mathbf{w}^{T}(\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}(y+\mu\mathbf{1}_{N})-\mathbf{V}_{0}^{-1}\mathbf{w}_0)
$$

Lastly, the terms that do not depend on $$\mathbf{w}$$ are grouped. 

$$
\mathbf{y}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{y}+\mu\mathbf{1}^{T}_{N}(\sigma^{2}\mathbf{I}_{N})^{-1}\mu\mathbf{1}_{N}-2\mu\mathbf{1}^{T}_{N}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{y}-\mathbf{w}_{0}^{T}\mathbf{V}_{0}^{-1}\mathbf{w}_{0} = constant
$$

Finally, the resulting expression is

$$
\exp{(-(2^{-1})(\mathbf{w}^{T}(\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{X} + \mathbf{V}_{0}^{-1})\mathbf{w}}-2\mathbf{w}^{T}((\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1})(y+\mu\mathbf{1}_{N})-\mathbf{V}_{0}^{-1}\mathbf{w}_0)+constant))
$$

Notice that this expression closely resembles the following quadratic form

$$
\mathbf{x}^{T}\mathbf{Ax} - 2\mathbf{x}^{T}\mathbf{Ad} + \mathbf{d}^{T}\mathbf{Ad} + e
$$

From this, it is straight forward to see that 

$$
\mathbf{A} = \mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}\mathbf{X} + \mathbf{V}_{0}^{-1}
$$

and

$$
\mathbf{Ad} = \mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}(\mathbf{y}+\mu\mathbf{1}_{N})-\mathbf{V}_{0}^{-1}\mathbf{w}_0
$$

the vector $$\mathbf{d}$$ is then 

$$
\mathbf{d} = \mathbf{A}^{-1}(\mathbf{X}^{T}(\sigma^{2}\mathbf{I}_{N})^{-1}(\mathbf{y}+\mu\mathbf{1}_{N})-\mathbf{V}_{0}^{-1}\mathbf{w}_0)
$$

if $$\mathbf{A}$$ is defined as $$\mathbf{V}_{N}^{-1}$$ and $$\mathbf{d}$$ is $$\mathbf{w}_{N}$$, then 

$$
\exp{(-(2^{-1})(\mathbf{w}^{T}\mathbf{V}_{N}^{-1}\mathbf{w} -2\mathbf{w}^{T}\mathbf{V}_{N}^{-1}\mathbf{w}_{N} + constant))}
$$

Recall that

$$
(\mathbf{x - d})^{T}\mathbf{A}(\mathbf{x - d}) + e = \mathbf{x}^{T}\mathbf{Ax} - 2\mathbf{x}^{T}\mathbf{Ad} + \mathbf{d}^{T}\mathbf{Ad} + e
$$

if the exponential operator is applied on the quadratic form, the constant term can be extracted out by the using the properties of exponentials as follows

$$
\exp{((\mathbf{x - d})^{T}\mathbf{A}(\mathbf{x - d}) + e )}
$$

$$
=\exp{((\mathbf{x - d})^{T}\mathbf{A}(\mathbf{x - d}))\exp{(e)}}
$$

because $$\exp{(e)}$$ is a constant, the quadratic exponential can be written as

$$
constant * \exp{((\mathbf{x - d})^{T}\mathbf{A}(\mathbf{x - d}))}
$$

This result is applied on the current form of the posterior 

$$
\exp{(-(2^{-1})(\mathbf{w}^{T}\mathbf{V}_{N}^{-1}\mathbf{w} -2\mathbf{w}^{T}\mathbf{V}_{N}^{-1}\mathbf{w}_{N} + constant))}
$$

$$
=\exp{(-(2^{-1})((\mathbf{w} - \mathbf{w}_{N})^{T}\mathbf{V}_{N}^{-1}(\mathbf{w} - \mathbf{w}_{N})+constant))}
$$

$$
= constant * \exp{(-(2^{-1})((\mathbf{w} - \mathbf{w}_{N})^{T}\mathbf{V}_{N}^{-1}(\mathbf{w} - \mathbf{w}_{N})))}
$$

and thus arriving at 

$$
p(\mathbf{w}|\mathbf{X, y}, \sigma{}^{2}) \propto  \exp{(-(2^{-1})((\mathbf{w} - \mathbf{w}_{N})^{T}\mathbf{V}_{N}^{-1}(\mathbf{w} - \mathbf{w}_{N})))} = \mathcal{N}(\mathbf{w}|\mathbf{w}_{N}, \mathbf{V}_{N})
$$
 
The next distribution of importance is the predictive distribution. This is of greater importance in machine learning because one is interested making predictions. This is where the Bayesian approach to linear regression differs from point estimates. Because the variance matrix is computed, the associated confidence intervals can be investigated. The posterior distribution at a test point is 

$$
p(y|\mathbf{x}, \mathbf{\mathcal{D}}, \sigma^{2}) = \int{\mathcal{N}(y|\mathbf{x}^{T}\mathbf{w}, \sigma^{2})\mathcal{N}(\mathbf{w}|\mathbf{w}_{N}, \mathbf{V}_{N})}d\mathbf{w}
$$

$$
=\mathcal{N}(y|\mathbf{w}_{N}^{T}\mathbf{x}, \sigma^{2}_{N}(\mathbf{x}))
$$

where

$$
\sigma^{2}(\mathbf{x}) = \sigma^{2} + \mathbf{x}^{T}\mathbf{V}_{N}\mathbf{x}
$$

Notice that the variance is a summation of two terms. The term on the left is attributed to noise in the observation and the remaining term is derived from variance in the parameters. Because the term on the right is multiplied by the observed data $$\mathbf{x}$$, it models the variance according to the distance of the observation and the training data. This will be demonstrated in the following section.

## Bayesian Linear Regression Application

The first step is to generate the synthetic data from a one-dimensional linear model. The linear model is of the form

$$
y(x, \mathbf{w}) = w_0 +w_{1}x + \epsilon
$$

where $$w_0 = -0.3$$ and $$w_1 = 0.5$$. To begin, the necessary libraries are imported.

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

#shortcut for distribution and bijector modules
tfd = tfp.distributions
tfb = tfp.bijectors
```
Next, a function is written to generate the synthetic data 

```python
def generate_data(w0, w1, samples, noise, seed=42, true_data=True):
    """Generate synthetic data from linear model."""
    #Initialize random number generator from a seed
    rng = tf.random.Generator.from_seed(seed)
    #Draw samples from uniform distribution 
    X = rng.uniform(minval=-1., maxval=1., shape=[samples, 1])
    X = tf.concat((tf.ones(shape=(samples, 1)), X), axis=1)
    # compute y = intercept + weight * data + noise
    y = w0 + w1 * X[:, 1:] + rng.normal(shape=[samples, 1], stddev=noise)
    if true_data:
        y_true = w0 + w1 * X[:, 1:] 
        return X, y, y_true
    else:
        return X, y

X, y, y_true = generate_data(-0.3, 0.5, 200, 0.2)
```

The generated data is visualized along with the true underlying linear model from which it was generated.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Dataset.html" height="525" width="100%"></iframe>

The modeling process begins by specifying the conjugate prior which is Gaussian. It is assumed that the Gaussian is centered at $$(0, 0)$$ and with variance $$\lambda^{2}\mathbf{I}$$. This is similar to ridge regression and thus $$\lambda$$ serves as a regularization parameter. 

```python
def gaussian_prior(lambda_,):
    """Define Multivariate Gaussian"""
    W_0 = tf.constant([[0.], [0.]])
    V_0 = (1. / lambda_) * tf.eye(2)
    prior = tfd.MultivariateNormalFullCovariance(loc=tf.squeeze(W_0), covariance_matrix=V_0)
    return prior, W_0, V_0

prior, W_0, V_0 = gaussian_prior(2.)
```
Because the prior variance is isotropic, the equiprobable contours will be circular. The prior can be visualized as

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Gaussian_prior.html" height="525" width="100%"></iframe>

The circular contours correspond to the first and second standard deviations. To begin understanding the underlying intuition of Bayesian linear regression, samples are drawn from the prior. The samples are then plotted in data and parameter space in the figure below.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Prior_samples.html" height="625" width="100%"></iframe>

Fifteen samples are drawn from the prior distribution and are displayed on the Gaussian contour subplot on the right. As expected, the samples are concentrated around regions of higher probability. When the samples are displayed in data space, they do not resemble the linear model from which the data was generated. This is expected because the prior has not been conditioned on the data. This is equivalent to asking a person to predict the parameters, $$w_0$$ and $$w_1$$ without seeing any data. The best they would be able to do is provide a random guess based upon any prior belief about a potential distribution (not necessarily a Gaussian distribution) about the parameters. 

If the same person observes one data point, they obtain more information and thus better parameter approximations could be provided. As they are exposed to incremental quantities of data, it is expected that the accuracy of their predictions increase. This is what will be observed when the posterior is computed because it is a combination of the prior and the likelihood which incorporates information derived from the observed data. To explore this idea, the posterior and posterior predictive distributions are defined as follows

```python
def posterior_dist(X, y, W_0, V_0, sigma_squared):
    #Posterior variance-covariance matrix
    V_N = sigma_squared * tf.linalg.inv(sigma_squared * V_0 + tf.transpose(X) @ X)
    #Posterior mean vector
    W_N = V_N @ V_0 @ W_0 + (1 / sigma_squared) * V_N @ tf.transpose(X) @ y
    #Posterior distribution
    posterior = tfd.MultivariateNormalFullCovariance(loc=tf.squeeze(W_N), covariance_matrix=V_N)
    return posterior, W_N, V_N

def posterior_predictive(l_bound, u_bound, sigma_squared, V_N, W_N):
    #Initialize data
    X = tf.linspace(l_bound, u_bound, 250)[:, tf.newaxis]
    X = tf.concat((tf.ones(shape=X.shape), X), axis=1)
    #Calculate variance
    variance = sigma_squared + tf.reduce_sum((X @ V_N * X), axis=1)
    stddev = tf.math.sqrt(variance)[:, tf.newaxis]
    y_pred = X @ W_N
    return  X, y_pred, y_pred + stddev, y_pred - stddev

observation = 1
sigma_squared = 0.5

posterior, W_N, V_N = posterior_dist(X[:observation], y[:observation], W_0, V_0, sigma_squared)
X_pred, y_pred, y_p_stddev, y_n_stddev = posterior_predictive(-2.5, 2.5, sigma_squared, V_N, W_N)
```

The posterior is computed by limiting the observed data to one data point and then be visualized on surface and contour plots displayed below

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Gaussian_posterior_1.html" height="525" width="100%"></iframe>

Note that the shape of the posterior has been altered by the inclusion of the likelihood. The equiprobable contours are ellipses. This is due to the additional information derived from the observed data. Given that only one data point has been observed, it is not likely that there will be a significant improvement when the posterior distribution is sampled. Fifteen samples are drawn from the posterior and displayed below. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Posterior_sample_1.html" height="525" width="100%"></iframe>

Note that the samples in data space do appear to be less random when compared to the samples from the prior. If the samples that contain a negative $$w_0$$ value and a positive $$w_1$$ value are counted, a total of six samples are observed compared to four from the prior distribution and which can be interpreted as a marginal improvement. Given that the posterior mean and variance have been computed, predictions can now be made on other inputs. The prediction mean is displayed below along with the intervals corresponding to two standard deviations of the posterior predictive density.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Predictive_posterior_1.html" height="525" width="100%"></iframe>

Note that the standard deviation decreases near the observed data and then increase further away from this point. This makes sense because one should be more confident in a prediction near observed data and less where no data has been observed. This process is repeated with 2, 20, and 150 data points.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Gaussian_posterior_2.html" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Posterior_sample_2.html" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Predictive_posterior_2.html" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Gaussian_posterior_3.html" height="525" width="100%"></iframe>

After observing twenty samples, the uncertainty associated with the parameters is diminishing thus shaping the Gaussian into a narrow peak about the expected parameters.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Posterior_sample_3.html" height="525" width="100%"></iframe>

This is also observed in the posterior predictive distribution. Notice that the standard deviation away from the prediction becomes narrow even as the predictions are more distant from the observed data.

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Predictive_posterior_3.html" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Gaussian_posterior_4.html" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Posterior_sample_4.html" height="525" width="100%"></iframe>

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/Predictive_posterior_4.html" height="525" width="100%"></iframe>

The intuition behind Bayesian linear regression is easy to comprehend because it is a natural way that one reasons with making predictions or drawing conclusions given data. Understanding simple examples of Bayesian inference provides a great way to explore more complex modeling methods because the results can be compared. This example was made possible because a conjugate prior was selected. If instead a non-conjugate prior was believed to describe the model parameters, an analytical solution could not be derived and thus requiring more advanced methods to determine the posterior and posterior predictive distributions. Monte Carlo methods are typically used when non-conjugate priors are selected. The following section replicates the results above using Hamiltonian Monte Carlo as explained on the Tensorflow website.

## Hamiltonian Markov Chain Monte Carlo

The modeling process begins by defining the joint posterior log probability.

```python
def posterior_log_prob(X_data, Y_data, offset, w, sigma):
    rv_sigma = tfd.Uniform(name="sigma", low=0., high=100.)
    rv_w = tfd.Normal(name="weights", loc=0., scale=5.)
    rv_offset = tfd.Normal(name="offset", loc=0., scale=5.)
    
    mu = offset + w * X_data
    rv_observed = tfd.Normal(name="obs", loc=mu, scale=sigma)
    
    return (
        rv_offset.log_prob(offset) + 
        rv_w.log_prob(w) + 
        rv_sigma.log_prob(sigma) +
        tf.reduce_sum(rv_observed.log_prob(Y_data))
    )
```

The next step is to define the initial conditions, the MCMC kernel, and then a tensorflow function to sample the chain.

```python
number_of_steps = 80000
burnin = 25000
step_size=tf.constant(0.5)

initial_chain_state = [
    tf.cast(1., dtype=tf.float32) * tf.ones([], name="init_offset", dtype=tf.float32),
    tf.cast(0.01, dtype=tf.float32) * tf.ones([], name="init_w", dtype=tf.float32),
    tf.cast(obs_sigma, dtype=tf.float32) * tf.ones([], name="init_sigma", dtype=tf.float32)
]

unconstraining_bijectors = [
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity()
]

unnormalized_posterior_log_prob = lambda *args: posterior_log_prob(X[:, 1], tf.squeeze(y), *args)

kernel = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            num_leapfrog_steps=2, 
            step_size=step_size,
            state_gradients_are_stopped=False
        ),
        num_adaptation_steps=int(burnin*0.8)
    ),
    bijector=unconstraining_bijectors
)

@tf.function
def run_chain(number_of_steps, burnin, initial_chain_state, kernel):
    chain_states, kernel_results = tfp.mcmc.sample_chain(
        num_results=number_of_steps,
        num_burnin_steps=burnin,
        current_state=(1., 0., obs_sigma),
        kernel=kernel, 
        name="HMC_sampling"
    )
    
    return chain_states, kernel_results
```

The chain can then be sampled by running the tensorflow funtiction `run_chain()`.

```python
chain_states, kernel_results = run_chain(number_of_steps, burnin, initial_chain_state, kernel)
```
The posterior distributions sampled from this model can then be visualized

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/BayLinReg/HMC_results.html" height="525" width="100%"></iframe>

The mean of each parameter is calculated as

```python
tf.reduce_mean(chain_states[0][burnin:]), \ #w_0
tf.reduce_mean(chain_states[1][burnin:]), \ #w_1
tf.reduce_mean(chain_states[2][burnin:])    #ε noise

(<tf.Tensor: shape=(), dtype=float32, numpy=-0.2877646>,
 <tf.Tensor: shape=(), dtype=float32, numpy=0.49686036>,
 <tf.Tensor: shape=(), dtype=float32, numpy=0.20349437>)
```

Notice that the true model parameters are within a $$5$$% error margin. The tools provided by Tensorflow and Tensorflow Probability are truly amazing for building probabilistic models and should be explored by data science and machine learning practitioners. More complex models will be explored  on future posts using Tensorflow and Tensorflow Probability on real world datasets. If there are any questions regarding the contents of this post please feel free to contact me via email and keep looking for my future posts.