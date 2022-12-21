---
layout: post
title:  "Implementing Metropolis-Hastings and Hamiltonian Monte Carlo on TensorFlow Probability"
date:   2022-12-19 -
description: "This post is the first in a series on Markov chain Monte Carlo. This is a tutorial on implementing the Metropolis-Hastings and Hamiltonian Monte Carlo algorithms using TensorFlow Probability. The main task is to estimate the parameters of a multivariate Gaussian distribution and estimate the posterior predictive distribution. This task was selected because it has a few difficulties that require solutions using TensorFlow Probability's available tools which can cause new users difficulties. Additionally, the analyical results can be compared to the MCMC computations to assure that the algorithms are working as intended."
categories: Markov_chain_Monte_Carlo TensorFlow_Probability Advanced_Bayesian_Computation 
html_url: /assets/img/MCMConTFP/Post and Obs Data.png
---

**Outline**
-   [Introduction](#introduction)
-   [Preliminaries](#preliminaries)
-   [Parameter Posterior Distribution Computation](#parameter-posterior-distribution-computation)
-   [Inverse-Wishart Distribution](#inverse-wishart-distribution)
-   [Parameter Posterior Distribution Computation Cont.](#parameter-posterior-distribution-computation-cont)
-   [Unconstraining Parameters](#unconstraining-parameters)
-   [Parameter Posterior Distribution Computation Cont. (Pt. 2)](#parameter-posterior-distribution-computation-cont-pt-2)
-   [Analytical Comparison and Posterior Predictive Distribution Computation](#analytical-comparison-and-posterior-predictive-distribution-computation)
-   [Conclusion](#conclusion)


## Introduction
---

Within the past few decades, several powerful methods have been developed that allow for approximating and simulating from complex probability distributions. These advanced computational methods can be implemented on available state-of-the-art platforms for statistical modeling and high-performance statistical computation making them practical for multiple applications. One of these methods is known as Markov chain Monte Carlo (MCMC). The Society for Industrial and Applied Mathematics (SIAM) has placed this method as one of top ten algorithms of the twentieth century and is paramount in Bayesian statistical analyses. This post is the first in a series that focuses on MCMC methods. Specifically, the implementation of the Metropolis-Hastings algorithm and Hamiltonian Monte Carlo using TensorFlow Probability (TFP). Subsequent posts will focus on how and why these methods work by constructing the algorithms using lower-level tools and analyzing the mathematical details.


The two MCMC methods will be implemented on data sampled from a multivariate normal (MVN) with known parameters with the objective of computing the parameter posterior and posterior predictive distributions. In completing these tasks, minor obstacles will be encountered and resolved by leveraging the powerful and flexible tools that TFP provides. Because the data is sampled from an MVN, the MCMC results will be compared with the analytical results to demonstrate that the algorithms work as intended. Lastly, it will be shown that the random walk behavior exhibited by the Metropolis- Hastings algorithm leads to computational inefficiencies that are remedied by incorporating gradient information of the posterior distribution as done in Hamiltonian Monte Carlo.

## Preliminaries
---

First, import the necessary libraries.

```python
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
import plotly.express as px
import arviz as 

tfd = tfp.distributions
tfb = tfp.bijectors
```

To generate the data, the mean vector and covariance matrix are specified. The precision matrix is also computed because it will become useful in this analysis. Lastly, the Cholesky factors are also computed to provide the correct parametrizations for TFP's multivariate normal distribution.

```python
true_mean = tf.zeros(2, dtype=tf.float32)
true_cov = tf.constant([[4., 1.8], [1.8, 1.]])
true_precision = tf.linalg.inv(true_cov)
true_cov_cholesky = tf.linalg.cholesky(true_cov)
true_precision_cholesky = tf.linalg.cholesky(true_precision)
data = tfd.MultivariateNormalTriL(loc=true_mean, 
                                  scale_tril=true_cov_cholesky).sample(100)
```

A scatterplot of the data is displayed below.

<img src="/assets/img/MCMConTFP/Observed Data.PNG" width="100%">


The data displays positive correlation and a larger degree of variance along the x-axis when compared to the y-axis as specified by the covariance matrix. With the data, the model for the posterior distribution of the parameters can be specified. 


## Parameter Posterior Distribution Computation
---

The first step in computing the posterior distribution of the parameters is to specify a model. TFP's **`JointDistributionSequential`** class can be used to model joint distributions as a collection of possibly interdependent distributions. Recall that the posterior parameter distribution can be written up to a constant of proportionality as

$$
\begin{align*}
\small p(\theta|y) & \propto  p(y|\theta) p(y) \\[1.5ex] 
\end{align*}
$$

One might define a model for the posterior distribution as

```python
posteriror = tfd.JointDistributionSequential([
    tfd.Distribution(parameters), #prior
    lambda prior_sample : tfd.Distribution(prior_sample) #likelihood
])
```

The user must specify the model by selecting adequate distributions for the prior and likelihood and their corresponding parameters. The appropriate conjugate distribution for multivariate normally distributed data with unknown mean and covariance is the normal-inverse-Wishart distribution. The model can then be written as

$$
\begin{align*}
\small p(\mu, \Sigma|y) & \propto  p(y|\mu, \Sigma) p(\mu, \Sigma) \\[1.5ex] 
& \; \small = \mathcal{N(y|\mu, \Sigma)} \; \text{NIW}(\mu, \Sigma | \mu_0, \kappa_0, \nu_0, \text{S}_0)
\end{align*}
$$

and in code 


```python
posterior = tfd.JointDistributionSequential([
    tfd.NormalInverseWishart(df=tf.constant(2.), counts=tf.constant(2.), 
                             loc=tf.zeors(2), scale_tril=tf.eye(2)), 
    lambda theta : tfd.MultivariateNormalTriL(loc=theta[0], scale_tril=theta[1])
])
```

Unfortunately, TFP does not have a normal-inverse-Wishart distribution. Recall that this distribution can be specified as a product of an inverse-Wishart and a conditional MVN distribution. Given this, the model is specified as

```python
posterior = tfd.JointDistributionSequential([
    tfd.InverseWishart(df=tf.constant(2.), scale_tril=tf.eye(2)), 
    lambda cov : tfd.MultivariateNormalTriL(loc=tf.zeros(2), 
                                            scale_tril=cov), 
    lambda mean, cov : tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
])
```

This code block executes as follows

1. Draw a 2x2 covariance matrix from an inverse-Wishart
2. Given the covariance matrix, draw a 2-dimensional mean vector 
3. Given the covariance matrix and mean vector, draw one 2-dimensional data vector


Note the structure of the last **`lambda`** function. In particular, note that the arguments are ordered in reverse from the corresponding sequential draws. When using the **`JointDistributionSequential`** class to design models, be aware of this structure as one can easily make a mistake with respect to the order of the arguments. Although an error would be generated given the dimensionality differences of the parameters in this example, when working with scalar data, no error would be generated thus resulting in an incorrect model. 

At this point, the first minor obstacle is encountered. TFP only has a Wishart distribution. Luckily, TFP has a module of bijective transformations that can be utilized to transform existing distributions into other distributions. To move forward with the posterior 
parameter distribution computation, an inverse-Wishart distribution is first created

## Inverse-Wishart Distribution
---

The **[distribution](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions)** that will be transformed is the **`WishartTriL`**, which is a Wishart distribution parameterized by Cholesky factors. Random matrices drawn from this distribution correspond to precision matrices, which are the inverse of covariance matrices. Thus, 

$$
\begin{align*}
\small \Sigma^{-1} = \Lambda \quad \text{where} \quad \Lambda \sim \text{Wi}(\nu, S)
\end{align*}
$$


Within TFP, this transformation can be done as 

```python
inverse_wishart = tfd.TransformedDistribution(
    distribution=tfd.WishartTriL(df=2., scale_tril=tf.eye(2)),
    bijector=tfb.MatrixInverse()
)
```

TFP's standard suite of **[bijectors](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors)** does not have a matrix inverse bijector; however, there is a **`CholeskyToInvCholesky`** bijector. To apply this transformation, the precision matrix drawn from the Wishart distribution must first be decomposed into its Cholesky factors. To achieve this a combination of bijectors will be used. These are the **`Invert`** and **`CholeskyOuterProduct`** bijectors. The **`CholeskyOuterProduct`** takes a Cholesky factor and computes the outer product. This transformation is shown below.

$$
\begin{align*}
\small L \rightarrow LL^{T} \quad \text{where} \quad LL^{T} = \Lambda 
\end{align*}
$$

Luckily, the **`Invert`** bijector inverts the transformation resulting in 


$$
\begin{align*}
\small L \leftarrow \Lambda
\end{align*}
$$

The chain of required transformations is


$$
\begin{align*}
\small \Lambda \rightarrow L \rightarrow L^{-1} \quad \text{where}\quad L^{-1}L^{-T} = \Sigma
\end{align*}
$$

The corresponding bijector is specified as 

```python
bijector = tfb.Chain([
    tfb.CholeskyToInvCholesky(),
    tfb.Invert(tfb.CholeskyOuterProduct())
])
```

The transformation can be verified by applying it to the **`true_precision`** variable which should equate to the **`true_cov_cholesky`** variable.

```python
print(bijector.forward(true_precision).numpy(), '\n', 
      true_cov_cholesky.numpy())

[[2.0000005 0.       ]
 [0.9000003 0.4358899]] 
 [[2.         0.        ]
 [0.9        0.43588996]]
```

The inverse-Wishart distribution is then specified. 

```python
inverse_wishart = tfd.TransformedDistribution(
    distribution=tfd.WishartTriL(df=2., scale_tril=tf.eye(2)),
    bijector=tfb.Chain([
        tfb.CholeskyToInvCholesky(),
        tfb.Invert(tfb.CholeskyOuterProduct())
    ])
)
```

Although this is an acceptable method of transforming a distribution, users may also create their own inverse-Wishart distribution class. This method holds many advantages over the method shown above. One can easily subclass TFP classes to reparametrize and transform existing distributions. The code below shows an inverse-Wishart distribution class created by subclassing TFP's **`TransformedDistribution`** class. 

```python
class InverseWishartTriL(tfd.TransformedDistribution):
    """Inverse Wishart parameterized by Cholesky factors"""
    
    def __init__(
        self, 
        df, 
        scale_tril, 
        name="InverseWishartTriL"
    ):
        parameters = dict(locals())
        super(InverseWishartTriL, self).__init__(
            distribution=tfd.WishartTriL(
                df=df,
                scale_tril=scale_tril,
                input_output_cholesky=False
            ), 
            bijector=tfb.Chain([
                tfb.CholeskyToInvCholesky(),
                tfb.Invert(tfb.CholeskyOuterProduct())
            ])
        )
        self._parameters = parameters     
        
    def _parameter_properties(self, dtype=tf.float32, num_classes=None):
        return dict(
            df = tfp.util.ParameterProperties(
                event_ndims=None, 
            ),
            scale_tril = tfp.util.ParameterProperties(
                event_ndims=None,
            )
        )  
```

With this new inverse-Wishart distribution, the modeling process can continue forward. 

## Parameter Posterior Distribution Computation Cont.
---

The code block below shows the specified model using the newly-created **`InverseWishartTriL`** class as well as a random draw from the model to assure that the model works correctly. 

```python
posterior = tfd.JointDistributionSequential([
    InverseWishartTriL(df=tf.constant(2.), scale_tril=tf.eye(2)),#covariance matrix
    lambda cov : tfd.MultivariateNormalTriL(loc=tf.zeros(2), 
                                            scale_tril=cov), #mean vector
    lambda mean, cov : tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
])

posterior.sample()

[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[ 1.3333945,  0.       ],
        [-1.6480633,  1.2821841]], dtype=float32)>,
 <tf.Tensor: shape=(2,), dtype=float32, numpy=array([-1.1494775,  2.5734134], dtype=float32)>,
 <tf.Tensor: shape=(2,), dtype=float32, numpy=array([-2.805678,  4.556411], dtype=float32)>]
```

With the model specified, it is a good moment to review the structure of the Metropolis-Hastings algorithm. The algorithm proceeds as follows

1. Begin at an initial state $$\small \theta_0$$ for which $$\small p(\theta_{0} \vert y) > 0$$
2. for $$\small t = 1, 2, ...$$ 

    a) sample a proposal $$\small \theta^*$$ from a proposal distribution  
    b) calculate the acceptance probability $$\small r=\text{min}\left(1, \frac{p(\theta^*|y)q(\theta^*|\theta^{t-1})}{p(\theta_0|y)q(\theta^{t-1}|\theta^*)}\right)$$ 
    
    c) set $$\small \theta^t = \theta^* \text{with probability min}(r, 1)$$ otherwise set $$\small \theta^t = \theta^{t-1}$$
    

TFP's **[MCMC](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc)** module contains a **`MetropolisHastings`** class that runs one step of the Metropolis-Hastings algorithm. The proposals generated by this class are generated from a symmetric distribution and thus the user only needs to specify the target log probability function, $$\small p(\theta \vert y)$$, which can be defined using the **`log_prob`** method from the model. Note that the target log probability function depends on the data and thus a simple call to the **`log_prob`** method is not sufficient. Instead, a new function is defined using a **`lambda`** function and the **`log_prob`** method. The new function is shown below.

```python
target_log_prob_fn = lambda *x : posterior.log_prob(list(x) + [data])
```

This function groups the arguments into a list then aggregates the data to this list and computes the log probability. Thus, if a covariance matrix and mean vector are passed as arguments, a list of the form **[covariance matrix, mean vector, data]** is expected. This is of the same form of the random draw that was observed when the **`sample`** method of the model was called except that the list contains the data at the end and not just one 2-dimensional vector. The target log probability function then evaluates the corresponding log probability of each argument under the specified distribution, aggregates them, and returns the corresponding value. Thus, it is expected that calling this function with one covariance matrix and mean vector sample, returns a single value. The code block below demonstrates what occurs when the function is called.

```python
cov, mean, _ = posterior.sample()
target_log_prob_fn(cov, mean)

<tf.Tensor: shape=(100,), dtype=float32, numpy=
array([  -6.30571  ,  -60.831585 ,  -27.849394 ,   -7.8490944,
        -26.134396 ,   -7.7007046,  -14.50868  ,   -8.001088 ,
         -6.3885956,   -6.6975546, -128.13167  ,  -60.49453  ,
         -6.88986  ,   -7.897996 ,  -23.344683 ,  -22.475151 ,
        -49.2268   ,  -37.13003  ,  -71.28161  ,  -26.67094  ,
         -7.5488634,  -18.85581  ,   -7.343771 ,  -25.666304 ,
        -16.773169 ,  -26.410807 ,  -10.495419 ,  -23.235426 ,
         -6.503666 ,  -12.970615 ,  -57.964638 ,  -74.973305 ,
        -15.89159  ,  -33.250618 ,   -8.169319 ,   -6.9937735,
        -19.309742 ,  -12.373661 ,  -46.474983 ,   -7.5679   ,
        -10.809265 ,   -9.131641 ,   -8.032692 ,  -14.9680805,
        -26.303654 ,   -9.390724 ,   -9.310343 ,   -9.725354 ,
        -11.996459 ,  -10.511663 ,   -7.272662 ,  -13.099302 ,
         -8.057295 ,  -50.57905  ,  -15.361183 ,   -7.1203704,
        -10.5487175,  -12.410334 ,   -7.155304 ,   -7.264439 ,
         -7.0043697,  -14.152628 ,  -12.47769  ,  -10.012243 ,
        -28.733065 ,  -12.731756 ,   -7.826803 ,   -9.207087 ,
         -9.115752 ,   -7.531613 ,   -7.1613383,  -37.65135  ,
        -22.394676 ,   -8.250543 ,   -6.6804905,  -17.98455  ,
        -16.124395 ,   -8.675791 ,   -8.387988 ,  -11.635809 ,
         -8.13366  ,   -6.8760023,  -10.037952 ,   -9.280255 ,
        -85.73548  ,   -6.2263117,  -41.657917 ,  -32.685745 ,
        -26.543512 , -100.13597  ,  -27.559698 ,  -32.79216  ,
         -7.4921637,  -42.878326 ,   -7.0890617,   -7.4344034,
        -17.668694 ,  -21.159586 ,   -6.445182 ,  -22.121353 ],
      dtype=float32)>
```

As observed, this does not occur.  The reason why this function is not returning a single log probability value is because the data likelihood was not specified to be i.i.d. and thus the data log probabilities cannot be summed. To specify that the data is i.i.d., TFD's **`Independent`** class can be applied. To fully understand how to use this class, a basic understanding of TensorFlows distribution shapes along with broadcasting rules are necessary. To keep focus on the implementation of the MCMC algorithms, these topics will not be expanded in this post but a great resource can be found **[here](https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes)**. 

The **`Independent`** class works by reinterpreting the rightmost batch dimension as part of the event dimension. The **`reinterpreted_batch_ndims`** parameter dictates the number of batch dimensions that are absorbed as event dimensions. The batch shape of the model is inspected below. 

```python
posterior.batch_shape

[TensorShape([]), TensorShape([]), TensorShape([])]
```

The three items in the list correspond to the batch shape of the covariance matrix, mean vector, and the data, respectively. All of the batches are empty and thus the batch shape needs to be augmented. TensorFlow shapes are of the form **[sample_shape, batch_shape, event_shape]**. The batch and event dimensions are provided by the specified distributions. In other words, at least the distribution of the data needs to be modified by expanding the left most dimension. This can be done easily by expanding the dimension of any of the two parameters (mean vector or covariance matrix) using **`tf.newaxis`** as follows


```python
posterior = tfd.JointDistributionSequential([
    InverseWishartTriL(df=tf.constant(2.), scale_tril=tf.eye(2)),
    lambda cov : tfd.MultivariateNormalTriL(loc=tf.zeros(2), 
                                            scale_tril=cov),
    lambda mean, cov : tfd.Independent(
        tfd.MultivariateNormalTriL(loc=mean[tf.newaxis, ...], scale_tril=cov),
        reinterpreted_batch_ndims=1
    )
])

posterior.batch_shape

[TensorShape([]), TensorShape([]), TensorShape([1])]
```

Note that this formulation of the model increases the previously empty batch dimension to by an additional unit. Setting the **`reinterpreted_batch_ndims`** parameter to one, absorbs the observed data matrix into a single event and thus the log probabilities are summed. The **`target_log_prob_fn`** function is called with the same covariance matrix and mean vector and a single log probability is returned.

```python
target_log_prob_fn(cov, mean)

<tf.Tensor: shape=(), dtype=float32, numpy=-460.46835>
```

Although this worked, expanding the dimension of the data distribution using **`tf.newaxis`** is brittle thus a generalized method is prefered. In fact, if the **`cov`** argument contains two or more samples, the log probability can no longer be evaluated due to incompatible dimensions. It is generally a good practice to allow for multiple batches of samples to be evaluated as it will be necessary when evaluating MCMC diagnostics. A better method uses TFP's **`BatchReshape`** class. This class takes an existing distribution and modifies is batch dimension to a specified vector. The final specified model is shown below. 

```python
posterior = tfd.JointDistributionSequential([
    tfd.BatchReshape(
        distribution=InverseWishartTriL(df=tf.constant(2.), scale_tril=tf.eye(2)),
        batch_shape=tf.constant([1])
    ), 
    lambda cov : tfd.MultivariateNormalTriL(
        loc=tf.zeros(2),
        scale_tril=cov
    ),
    lambda mean, cov : tfd.Independent(
        distribution=tfd.MultivariateNormalTriL(
            loc=mean, 
            scale_tril=cov
        ),
        reinterpreted_batch_ndims=1
    )
])
```

Because the **`cov`** sample contains the batch and event dimensions, it transfers this information to the subsequent dependent distributions and thus they do not to be modified if the batch dimension is the same. Now, if multiple samples are to be evaluated, a single corresponding log probability should be returned when calling the **`target_log_prob_fn`**. This is verified below. 

```python
cov, mean, _ = model.sample(4)
target_log_prob_fn(cov, mean)

<tf.Tensor: shape=(4,), dtype=float32, numpy=array([ -625.811  , -1217.3715 , -1625.6641 ,  -599.54047], dtype=float32)>
```

With the model and target log probability function specified, the Metropolis-Hastings algorithm can be implemented. To use the **`MetropolisHastings`** class, the **`inner_kernel`** argument must be specified. This parameter takes an uncalibrated transition kernel and uses it to sample from the target distribution. In this example, the **`UncalibratedRandomWalk`** is used as follows

```python
kernel = tfp.mcmc.MetropolisHastings(
    inner_kernel=tfp.mcmc.UncalibratedRandomWalk(
        target_log_prob_fn=target_log_prob_fn
    )
)
```

The **`kernel`** variable contains a method that runs one step of the Metropolis-Hastings algorithm. The method requires that the **`current_state`** and **`previous_kernel_results`** parameters be specified and it returns a tuple with the current state and the corresponding results. The results are shown individually below. 

```python
cov, mean, _ = posterior.sample()
init_state = [cov, mean]

one_step = kernel.one_step(init_state, kernel.bootstrap_results(init_state))
for i in range(len(one_step[1])):
    print(i, one_step[1][i], '\n')
    
0 UncalibratedRandomWalkResults(
  log_acceptance_correction=<tf.Tensor: shape=(), dtype=float32, numpy=0.0>,
  target_log_prob=<tf.Tensor: shape=(), dtype=float32, numpy=-487.66296>,
  seed=[]
) 

1 tf.Tensor(False, shape=(), dtype=bool) 

2 tf.Tensor(-1232.7152, shape=(), dtype=float32) 

3 [<tf.Tensor: shape=(1, 2, 2), dtype=float32, numpy=
array([[[2.0877342 , 0.09860282],
        [0.35491884, 0.41267812]]], dtype=float32)>, <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[-1.9604735,  1.7822126]], dtype=float32)>] 

4 UncalibratedRandomWalkResults(
  log_acceptance_correction=<tf.Tensor: shape=(), dtype=float32, numpy=0.0>,
  target_log_prob=<tf.Tensor: shape=(), dtype=float32, numpy=-1720.3782>,
  seed=<tf.Tensor: shape=(2,), dtype=int32, numpy=array([ 2109161949, -2010540171])>
) 

5 [] 

6 tf.Tensor([-1106154913  1329956691], shape=(2,), dtype=int32) 
```

These results are summarized as 

0. Accepted results -  log acceptance correction factor, target log probability, and seed  
of the accepted state.
1. Proposal acceptance - Boolean that indicates if proposal was accepted
2. Log acceptance probability ratio - 
$$\small r = \text{min}\left(1, \frac{p(\theta^*|y)q(\theta^*|\theta^{t-1})}{p(\theta_0|y)q(\theta^{t-1}|\theta^*)}\right)$$
3. Proposed state - proposed state generated by the sampler
4. Proposed state results - proposed state results
5. Space
6. Seed


This process is repeated $$\small n$$ times, resulting in a Markov chain whose states correspond to values sampled from the posterior distribution. To run the Metropolis-Hastings algorithm for multiple steps, the **`sample_chain`** function in TFP's MCMC module can be used. The signature of the **`sample_chain`** function is displayed below.



```python
tfp.mcmc.sample_chain(
    num_results,
    current_state,
    previous_kernel_results=None,
    kernel=None,
    num_burnin_steps=0,
    num_steps_between_results=0,
    trace_fn=<function <lambda> at 0x0000022A2710CB80>,
    return_final_kernel_results=False,
    parallel_iterations=10,
    seed=None,
    name=None,
)
```

The parameters that need to be specified are **`num_results`**, **`current_state`**, **`kernel`**,  **`num_burnin_steps`**, **`num_steps_between_results`**, and **`trace_fn`**. Given these parameters, the function returns the states and the corresponding kernel results. This function can be implemented as follows

```python
@tf.function(autograph=False, jit_compile=True)
def sample(num_results, current_state, kernel, num_burnin_steps, num_steps_between_results):
    
    num_results = tf.constant(num_results)
    num_burnin_steps = tf.constant(num_burnin_steps)
    num_steps_between_results = tf.constant(num_steps_between_results)
    
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=kernel,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results,
        trace_fn=lambda current_state, kernel_results : kernel_results,
        )

cov, mean , _ = posterior.samle()
init_state = [cov, mean]
kernel = tfp.mcmc.MetropolisHastings(
    inner_kernel=tfp.mcmc.UncalibratedRandomWalk(
        target_log_prob_fn=target_log_prob_fn
    )
)

num_results = 1000
num_burnin_steps = 200
num_steps_between_results = 150
states, kernel_results = sample(num_results, init_state, kernel, 
                                num_burnin_steps, num_steps_between_results)
```

Although the Metropolis-Hastings algorithm runs as intended, a critical detail was overlooked resulting in an inadequate posterior parameter distribution. The traceplots of the posterior samples are displayed below. Notice the large periods in which the trace plots remain flat. This is due to proposed values being rejected and thus the new state is set to the previous state. This immediately raises questions as to why the sampler is proposing a large number of proposals that are rejected. This is another minor obstacle that needs to be addressed. 

<img src="/assets/img/MCMConTFP/Traceplot.PNG" width="100%">

## Unconstraining Parameters
---

Recall that covariance matrices are symmetric and positive definite. The sampler proposes a new state by adding gaussian noise to the previous state, thus resulting in a matrix that is highly likely to not be symmetric and positive definite. The proposed matrix thus has a low probability of being a valid covariance matrix that describes the data and is rejected. The first proposed state by the sampler can be found within the variable **`kernel_results`** and is shown below. The first proposal fails the constrain imposed by covariance matrices. 

```python
print(kernel_results.proposed_state[0][0])

tf.Tensor(
[[[ 13.551279    2.3483853]
  [116.806915   13.927662 ]]], shape=(1, 2, 2), dtype=float32)
```

It is not uncommon to encounter constrained parameters given that different distributions are only valid within a finite support. For example, the uniform distribution describes random variables that are constrained to an interval. To unconstrain the random variables, the interval of support is mapped to the real numberline. In this example, covariance matrices need to be mapped into vectors with unconstrained entries. A great resource for unconstraining parameters is **[Stan's](https://mc-stan.org/docs/reference-manual/variable-transforms.html)** language reference manual. The following bijector transforms covariance matrices into unconstrained vectors.

```python
bijector=tfb.Chain([
    tfb.Invert(tfb.FillScaleTriL()),
    tfb.CholeskyToInvCholesky(),
    tfb.Invert(tfb.CholeskyOuterProduct())
])

print(bijector.forward(true_covariance))

tf.Tensor([ 0.54130864 -2.0647414   0.76497847], shape=(3,), dtype=float32)
```

With this bijective transformation, a new unconstrained inverse-Wishart distribution class can be created.


```python
class UnconstrainedInverseWishartTriL(tfd.TransformedDistribution):
    """Unconstrained Inverse Wishart parameterized by Cholesky factors"""
    def __init__(
        self, 
        df, 
        scale_tril, 
        name="UnconstrainedInverseWishartTriL"
    ):
        parameters = dict(locals())
        super(UnconstrainedInverseWishartTriL, self).__init__(
            distribution=tfd.WishartTriL(
                df=df,
                scale_tril=scale_tril,
                input_output_cholesky=False
            ), 
            bijector=tfb.Chain([
                tfb.Invert(tfb.FillScaleTriL()),
                tfb.CholeskyToInvCholesky(),
                tfb.Invert(tfb.CholeskyOuterProduct())
            ])
        )
        self._parameters = parameters     
        
    def _parameter_properties(self, dtype=tf.float32, num_classes=None):
        return dict(
            df = tfp.util.ParameterProperties(
                event_ndims=None, 
            ),
            scale_tril = tfp.util.ParameterProperties(
                event_ndims=None,
            )
        )  
```

The corresponding modified model is shown below

```python
bijector_tril = tfb.FillScaleTriL()

posterior = tfd.JointDistributionSequential([
    tfd.BatchReshape(
        distribution=UnconstrainedInverseWishartTriL(df=tf.constant(2.), 
                                                     scale_tril=tf.eye(2)),
        batch_shape=tf.constant([1])
    ), 
    lambda cov : tfd.MultivariateNormalTriL(
        loc=tf.zeros(2),
        scale_tril=bijector_tril.forward(cov)
    ),
    lambda mean, cov : tfd.Independent(
        distribution=tfd.MultivariateNormalTriL(
            loc=mean, 
            scale_tril=bijector_tril.forward(cov)
        ),
        reinterpreted_batch_ndims=1
    )
])

target_log_prob_fn = lambda *x : posterior.log_prob(list(x) + [data])
```

Because the **`UnconstrainedInverseWishartTriL`** now returns a vector, the corresponding draws must first be transformed into Cholesky factors before being passed as distribution parameters. With the unconstrained parameterization the posterior parameter distribution can be computed. 

## Parameter Posterior Distribution Computation Cont. (Pt. 2)
---

The **`sample`** function is then called and the traceplot of the posterior samples is displayed  
below.

```python
cov, mean , _ = posterior.sample()
init_state = [cov, mean]
kernel = tfp.mcmc.MetropolisHastings(
    inner_kernel=tfp.mcmc.UncalibratedRandomWalk(
        target_log_prob_fn=target_log_prob_fn
    )
)
num_results = 10000
num_burnin_steps = 2000
num_steps_between_results = 1500
states, kernel_results = sample(num_results, init_state, kernel, 
                                num_burnin_steps, num_steps_between_results)

posterior_cov = bijector_tril.forward(states[0])
posterior_cov = tf.matmul(posterior_cov, posterior_cov, transpose_b=True)
posterior_mean = states[1]
```

<img src="/assets/img/MCMConTFP/Unconstrained Traceplot.PNG" width="100%">


Finally, the posterior parameter distributions are computed. Before moving forward with the analytical comparisons, note that the **`num_steps_between_results`** parameter is set to $$\small 1500$$. This means that a total of fifteen-hundred samples are discarded before a sample is considered as a proposal. This is done to increase the exploration of the posterior parameter space. This inefficiency is attributed to the random walk behavior of the Metropolis-Hastings algorithm. The application of this algorithm on complex problems is not practical and thus a better method is required. Hamiltonian Monte Carlo (HMC) does not exhibit random walk behavior making far more efficient and practical than the Metropolis-Hastings algorithm. HMC incorporates information about the gradient of posterior distribution and thus explores parameter space efficiently. The signature of TFP's **`HamiltonianMonteCarlo`** class is shown below.

```python
tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn,
    step_size,
    num_leapfrog_steps,
    state_gradients_are_stopped=False,
    step_size_update_fn=None,
    store_parameters_in_results=False,
    experimental_shard_axis_names=None,
    name=None,
)
```

To implement HMC, only the **`target_log_prob_fn`**, **`step_size`**, and **`num_leapfrog_steps`** need to be defined. The same log probability function that was used for the Metropois-Hastings algorithm can be used for this kernel and thus only the step size and number of leapfrog steps need to be defined. These parameters are crucial for the efficient exploration of the parameter space and their optimal values are difficult to find. In practice, they must be tuned in an iterative manner. TFP's mcmc module does have multiple step size adaptation kernels that will adapt the step size according to some policy. The **`SimpleStepSizeAdaptation`** will be used to adapt the step size and its signature is shown below.

```python
tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel,
    num_adaptation_steps,
    target_accept_prob=0.75,
    adaptation_rate=0.01,
    step_size_setter_fn=<function hmc_like_step_size_setter_fn at 0x0000022A270B48B0>,
    step_size_getter_fn=<function hmc_like_step_size_getter_fn at 0x0000022A270B4E50>,
    log_accept_prob_getter_fn=<function hmc_like_log_accept_prob_getter_fn at 0x0000022A270BD0D0>,
    reduce_fn=<function reduce_logmeanexp at 0x0000022A269441F0>,
    experimental_reduce_chain_axis_names=None,
    validate_args=False,
    name=None,
)
```

The main parameters to specify are **`num_adaptation_steps`** and **`target_accept_prob`**. The target accept probability varies for different MCMC algorithms but for HMC, an acceptable range is within $$\small 0.6 \text{ and } 0.9$$. The number of adaptation steps is typically set to a value smaller than the number of burnin steps. Users must inspect the behavior of the chain during the burnin phase to determine if a sufficient number of steps were allotted for adequate adaptation. Recall that in the previous model, an unconstrained parameterization of the inverse-Wishart distribution was required. Instead of that approach, TFP also has a **`TransformedTransitionKernel`** class that carries out the task of taking unconstrained parameters and transforming them to the constrained parameter so that the log probability is properly calculated. The HMC kernel is shown below.

```python
posterior = tfd.JointDistributionSequential([
    tfd.BatchReshape(
        distribution=InverseWishartTriL(df=tf.constant(2.), scale_tril=tf.eye(2)),
        batch_shape=tf.constant([1])
    ), 
    lambda cov : tfd.MultivariateNormalTriL(
        loc=tf.zeros(2),
        scale_tril=cov
    ),
    lambda mean, cov : tfd.Independent(
        distribution=tfd.MultivariateNormalTriL(
            loc=mean, 
            scale_tril=cov
        ),
        reinterpreted_batch_ndims=1
    )
])

target_log_prob_fn = lambda *x : posterior.log_prob(list(x) + [data])

bijector=[
    tfb.FillScaleTriL(), #transforms unconstrained vector to Cholesky factor
    tfb.Identity() #mean vector has support over the entire real numberline
]

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=0.1,
            num_leapfrog_steps=10,
        ), 
        bijector=bijector
    ),
    num_adaptation_steps=int(num_burnin_steps*0.8),
    target_accept_prob=0.65,
)
```

The **`bijector`** variable is a list that contains the constrained transformations. They are applied according to the structure of the specified model(note that the unconstrained parameterization of the inverse-Wishart is no longer being used in the model). The first bijector takes a vector of reals and transforms it to a Cholesky factor which the model can then calculate the log probability. Because there are two states that are being evaluated, the **`TransformedTransitionKernel`** expects two bijectors. Given that the multivariate normal distribution is unconstrained, an identity transformation is provided. The step size and number of leapfrog steps are set to arbitrary values for the initial run. Lastly, eighty percent of the burnin phase is allotted for adaptation and the target acceptance probability is set within the appropriate range for HMC. The chain is then sampled and the traceplot is displayed below.

```python
cov, mean , _ = posterior.sample()
init_state = [cov, mean]
num_results = 6000 
num_burnin_steps = 2000
num_steps_between_results = 0

states, kernel_results = sample(num_results, init_state, kernel, num_burnin_steps, 
                                num_steps_between_results)
```

<img src="/assets/img/MCMConTFP/HMC Traceplot.PNG" width="100%">

Note that both the KDE estimates and traceplots look substantially better than those generated by the Metropolis-Hastings algorithm. This was achieved with only $$\small 4,000$$ samples in comparison to the $$\small 10,000$$ samples drawn when the Metropolis-Hastings algorithm was implemented. This is a direct effect of the efficiency of HMC. The results can then be compared to the analytical solutions to demonstrate that the resulting draws do belong to the desired posterior distribution. 

## Analytical Comparison and Posterior Predictive Distribution Computation
---

It can be shown that the posterior is a normal-inverse-Wishart distribution with updated parameters:  

$$
\begin{align*}
\small p(\mu, \Sigma|y) & \; \small = \text{NIW}(\mu, \Sigma|\mu_N, \kappa_N, \nu_N, \text{S}_N) \\[1.5ex] 
\small \mu_N & = \small \frac{\kappa_0 \mu_0 + N \bar{y}}{\kappa_0 + N} \\[1.5ex]
\small \kappa_N & \; \small = \kappa_0 + N \\[1.5ex]
\small \nu_N & \; \small = \nu_0 + N \\[1.5ex]
\small \text{S}_N & \; \small =  \text{S}_0 + \text{S}_{\bar{y}} + \frac{\kappa_0 N}{\kappa_0 + N}(\bar{y} - \mu_0)(\bar{y} - \mu_0)^{T}
\end{align*}
$$
 
Analogous to the multivariate results above, the univarte posterior is a normal-inverse-chi-squared with updated parameters. 

$$
\begin{align*}
\small p(\mu, \Sigma|y) & \; \small = \text{NI}\chi^{2}(\mu, \sigma^{2}|\mu_N, \kappa_N, \nu_N, \sigma^2_N) \\[1.5ex] 
\small \mu_N & = \small \frac{\kappa_0 \mu_0 + N \bar{y}}{\kappa_0 + N} \\[1.5ex]
\small \kappa_N & \; \small = \kappa_0 + N \\[1.5ex]
\small \nu_N & \; \small = \nu_0 + N \\[1.5ex]
\small \nu_N\sigma^2_N  & \; \small = \nu_0\sigma^2_0 + \sum_{i=1}^{N}(y_i-\bar{y})^2 + \frac{N\kappa_0}{\kappa_0+N}(\mu_0-\bar{y})^2
\end{align*}
$$

And the marginals can be shown to be 

$$
\begin{align*}
\small p(\sigma^2|y) & \; \small = \int p(\mu, \sigma^2|y)d\mu = \chi^{-2}(\sigma^2|\nu_N, \sigma_N^2) \\[1.5ex]
\small p(\mu|y) & \; \small = \int p(\mu, \sigma^2|y)d\sigma^2 = \mathcal{T}(\mu|\mu_N, \frac{\sigma_N^2}{\kappa_N}, \nu_N)
\end{align*}
$$


With the following prior parameters, the marginals can then be specified.

```python 
kappa_0 = 0
mu_0 = 0
N = 100
y_bar = tf.reduce_mean(data, axis=0)
nu_0 = -1
nu_N_sigma_N_squared = tf.transpose(data) @ data
sigma_N_squared = nu_N_sigma_N / nu_N
mu_N = (kappa_0 * mu_0 + N * y_bar) / (kappa_0 + N)
kappa_N = kappa_0 + N
nu_N = nu_0 + N

student_t = tfd.StudentT(df=nu_N, loc=mu_N[i], scale=tf.sqrt(sigma_N_squared[i, i]/kappa_N))
#Scaled inverse chi squared as a reparameterized inverse gamma
inverse_chi_squared = tfd.InverseGamma(nu_N/2, nu_N_sigma_N_squared[i, j]/2)
```

The HMC samples are compared to the marginal distributions in the image below.

<img src="/assets/img/MCMConTFP/HMC and Analytical Post.PNG" width="100%">

The analytical and HMC samples are in agreement thus the HMC samples are drawn from the posterior parameter distribution. The final task is to draw samples from the posterior predictive distribution and compare them to the corresponding analytical results. The posterior predictive distribution samples can be drawn by calling the model's **`sample`** method and passing in a list of the posterior parameters as arguments into the **`value`** parameter. This procedure is shown below. 


```python
_, _, posterior_samples = posterior.sample(value=[states[0], posterior_mean, None])
```

The posterior predictions are first sampled from the model and are plotted against the data.

<img src="/assets/img/MCMConTFP/Post and Obs Data.PNG" width="100%">

This figure demonstrates that the observed data is bounded by the posterior prediction samples. The posterior predictive samples follow a multivariate student-t distribution and the marginalized samples follow univariate student-t distributions and are parameterized as 

$$
\begin{align*}
\small p(\tilde{y}|y) & \; \small = \int \int p(\tilde{y}|\mu, \sigma^2)p(\mu, \sigma^2|y)d\mu d\sigma^2= \mathcal{T}(\tilde{y}|\mu_N, \frac{(1 + \kappa_N)\sigma_N^2}{\kappa_N}, \nu_N)
\end{align*}
$$

Given this, the marginal posterior predictive distribution can be specified as

```python
scaled_sigma_N = tf.sqrt((1 + kappa_N) / kappa_N * sigma_N_squared)
student_t = tfd.StudentT(df=nu_N, loc=mu_N[i], scale=scaled_sigma_N[i, i])
```

The comparison between the HMC marginal predictive samples and the analytical results are displayed below

<img src="/assets/img/MCMConTFP/Marginal Posterior Predictive.PNG" width="100%">

These results are in agreement and thus it can be concluded that the parameter and predictive distributions are correctly approximated by the HMC method. 

## Conclusion
---

TFP provides the necessary tools to construct expressive probabilistic models that can be used for making probabilistic statements on parameters and on predictions. This example demonstrated the application the Metropolis-Hastings and Hamiltonian Monte Carlo algorithms using TFP's tools. In the application of these methods, it was demonstrated how to build new distributions that are not within TFP's standard suite of distributions using bijective transformations. Additionally, the available bijective transformations were applied to unconstrain parameters. Although the Metropolis-Hastings worked in approximating the posterior parameter distribution, when compared to Hamiltonian Monte Carlo, it is far less efficient and thus not practical in applications that are more complex. Users new to TFP can build on these examples to tackle on more complex tasks and avoid or quickly remedy common errors made when building models in TFP. 