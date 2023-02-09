---
layout: post
title:  "Clustering with Gaussian Mixture Models"
date:   2023-02-07 -
description: "This is a brief demonstration on clustering using Gaussian Mixture Models on Tensorflow_Probability"
categories: Markov_chain_Monte_Carlo Latent_Variable_Models 
html_url: /assets/img/GMMClustering/Clustered Scatterplot Matrix.png
---

**Outline**
-   [Introduction](#introduction)
-   [Data Loading and Visualization](#data-loading-and-visualization)
-   [Model Specification and Sampling](#model-specification-and-sampling)
-   [Responsibility Calculation and Label Assignment](#responsibility-calculation-and-label-assignment)
-   [Conclusion](#conclusion)

## Introduction
---

This is a brief example that demonstrates how to cluster unlabeled data using a Gaussian mixture model (GMM). This will be demonstrated on a dataset that contains user data gathered by an online advertising agency. For simplicity, the original dataset has been limited to three features namely, the daily time a user spends on the site, the user's area income, and the daily internet usage of the user. The procedure begins by fitting a GMM to the data using Hamiltonian Monte Carlo (HMC). Once the GMM's parameters are inferred, the posterior probability that a particular user belongs to one of the assumed number of cluster is calculated. The posterior probabilities will be used to cluster the data into two groups and then will be compared to the original labels. 

## Data Loading and Visualization
---

The first step is to import the necessary libraries.

```python
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

tfd = tfp.distributions
tfb = tfp.bijectors
```

Next, the data is loaded and processed. 

```python
#cast data into a tensor and standardize
X = tf.cast(tf.constant(df.to_numpy()[:, :-1]), tf.float32)
y = tf.cast(tf.constant(df.to_numpy()[:, -1:4]), tf.float32)
mu_hat = tf.reduce_mean(X, axis=0)
sigma_hat = tfp.stats.covariance(X)
#diagonal entries correspond to feature variances
X_std = (X - mu_hat) / tf.sqrt(tf.linalg.diag_part(sigma_hat))
#concat features with labels
X = tf.concat([X, y], axis=1)
X_std = tf.concat([X_std, y], axis=1)
```

A good way of visualizing numeric data is to plot on a scatterplot matrix with the associated histogram along the diagonal entries. This may give an analyst insight into any potential correlation of the data and how it's distributed. 

<img src="/assets/img/GMMClustering/Scatterplot Matrix.png" width="100%">

The scatterplots and the histograms display that there may be two potential clusters. Given that the original goal of the advertising agency was to determine whether users clicked or did not click on the advertisement, the GMM will be implemented under the assumption that there are two clusters.

## Model Specification and Sampling
---

Clustering data requires computing the posterior probability that a point $$i$$ belongs to a cluster $$k$$. This is known as the **responsibility** of cluster $$k$$ for the point $$i$$ and is defined as follows:

$$
\small r_{ik} = p(z_i=k \vert \mathbf{x}_i, \theta) = \frac{p(\mathbf{x}_i \vert z_i=k, \theta)p(z_i=k)}{\sum_{k^{'}=1}^{K}p(\mathbf{x}_i \vert z_i=k^{'}, \theta)p(z_i=k^{'})}
$$

 For clustering, categorical and Gaussian distributions are used to specify the prior and the likelihood terms, respectively. Thus the numerator of the responsibility has the form of

$$
\sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i \vert \mu_k, \Sigma_k)
$$

Using a Bayesian approach to compute the posterior, the following model is constructed.

$$
\theta \sim \text{Dir}(\alpha) \\[1.5ex]
\mu_k \sim \mathcal{N}(\mu_o, \Sigma_o) \\[1.5ex]
\Sigma_k \sim \text{IW}(S, \nu) \\[1.5ex]
\pi_k \sim \text{Cat}(\theta) \\[1.5ex]
\mathbf{x}_i \sim \mathcal{N}(\mu_k, \Sigma_k)
$$

The code block below is the corresponding joint distribution and the target log probability function.

```python
def joint_distribution(concentration, mu, sigma, dof, scatter_tril, sample_shape):
    model = tfd.JointDistributionSequential([
    tfd.Independent(
        tfd.BatchReshape(
            tfd.Dirichlet(
                concentration=concentration
            ),
            batch_shape=tf.constant([1])
        ), 
        reinterpreted_batch_ndims=1
    ),
    tfd.Independent(
        tfd.BatchReshape(
            tfd.Normal(
                loc=mu, 
                scale=sigma
            ), 
            batch_shape=tf.constant([2, 3])
        ), 
        reinterpreted_batch_ndims=2
    ),
    tfd.Independent(
        tfd.BatchReshape(
            tfd.Sample(
                InverseWishartTriL(
                    df=dof,
                    scale_tril=scatter_tril
                ),
                sample_shape=2
            ),
            batch_shape=tf.constant([1])
        ), 
        reinterpreted_batch_ndims=1
    ),
    lambda scale_tril, loc, probs : tfd.Independent( 
        tfd.Sample(
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=probs),
                components_distribution=tfd.MultivariateNormalTriL(
                    loc=loc, 
                    scale_tril=scale_tril
                )
            ),
            sample_shape=sample_shape
        ),
        reinterpreted_batch_ndims=1
    )
])
    
    return model

clusters = 2
features = 3
samples = 1000
model = joint_distribution(tf.ones(clusters), tf.zeros((clusters, features)), 
                           tf.constant(1.), 5, tf.eye(features), samples)

target_log_prob_fn = lambda *x : model.log_prob(list(x) + [X_std[:, :-1]])
```

The next step is to specify the Markov chain and draw samples from the posterior.

```python
chains = 1
probs, loc, scale_tril, _ = model.sample(chains)
current_state = [probs, loc, scale_tril]
num_results = 8000
num_burnin_steps = 2000
num_steps_between_results = 0
num_leapfrog_steps = 5

bijector = [
    tfb.SoftmaxCentered(),
    tfb.Identity(),
    tfb.FillScaleTriL()
]

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=[0.015 for _ in bijector],
            num_leapfrog_steps=num_leapfrog_steps,
        ),
        bijector=bijector
    ), 
    num_adaptation_steps=int(num_burnin_steps*0.8),
    target_accept_prob=0.65,
    adaptation_rate=0.01,
    reduce_fn=tfp.math.reduce_log_harmonic_mean_exp
)

@tf.function(autograph=False, jit_compile=True)
def sample(num_results, current_state, kernel, num_burnin_steps, num_steps_between_results):
    
    num_results = tf.constant(num_results)
    num_burnin_steps = tf.constant(num_burnin_steps)
    num_steps_between_results = tf.constant(num_steps_between_results)
    
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=current_state,
        kernel=kernel,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results,
        trace_fn=lambda current_state, kernel_results : kernel_results,
        )

#sample posterior

states, kernel_results = sample(num_results, current_state, kernel, num_burnin_steps, 
                                num_steps_between_results)
```

The traceplot shown below suggests that the Markov chain does converge to the target distribution and thus the posterior samples can be used to calculate the associated responsibilities.

<img src="/assets/img/GMMClustering/traceplot.png" width="100%">

## Responsibility Calculation and Label Assignment
---

The responsibilities are calculated below using the median and the data is clustered. 

```python
#parameter quantiles calculations
probs_quantiles = tfp.stats.percentile(states[0], [2.5, 50., 97.5], axis=0)
mean_quantiles = tfp.stats.percentile(states[1], [2.5, 50., 97.5], axis=0)
tril_quantiles = tfp.stats.percentile(states[2], [2.5, 50., 97.5], axis=0)

#likelihood is evaluated using the median from the posterior parameter distribution
likelihood_log_prob = tfd.MultivariateNormalTriL(loc=mean_quantiles[1], 
                                                 scale_tril=tril_quantiles[1]).log_prob
#log p(z_i = k|theta) + p(x_i|z_i = k, theta)
class_log_prob = likelihood_log_prob(tf.expand_dims(X_std[:, :-1], axis=1)) + \
                      tf.math.log(probs_quantiles[1])
#log Sum_{k=1}^{K} p(z_i = k'|theta) + p(x_i|z_i = k', theta)
marginal_log_prob = tfp.math.log_add_exp(class_log_prob[0, :, 0], 
                    class_log_prob[0, :, 1])[..., tf.newaxis]
#log p(z_i = k|x_i, theta)
log_responsibilities = class_log_prob - marginal_log_prob
#label each user to a cluster
clusters = tf.where(log_responsibilities[0, :, 0] < log_responsibilities[0, :, 1], 1, 0)
```
The images below display the resulting clusters and the true clusters associated with the data. 

<img src="/assets/img/GMMClustering/Clustered Scatterplot Matrix.png" width="100%">

<img src="/assets/img/GMMClustering/True Clusters.png" width="100%">