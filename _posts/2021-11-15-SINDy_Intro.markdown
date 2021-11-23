---
layout: post
title:  "Introduction to Sparse Identification of Nonlinear Dynamics"
date:   2021-11-16 -
description: "The sparse identification of nonlinear dynamics (SINDy) is a method used to identify governing equations from dynamical systems using measurement data. This method relies on the assumption that many dynamical systems have few contributing terms that exist within high-dimensional nonlinear function space. The SINDy algorithm is applied on the Lorenz system of equations to demonstrate its functionality."
categories: Python Sparse_Identification_of_Nonlinear_Dynamics Data-Driven_Methods Sparse_Regression 
html_url: /assets/img/SINDy/Lorenz Attractor.html
---

**Outline**
-   [Overview](#overview)
-   [Lorenz System Application](#lorenz-system-application)
-   [Python Code Overview](#python-code-overview)

## Overview

Understanding the characteristics of dynamical systems has been of major interest across multiple disciplines in science, mathematics, and engineering. Although many governing equations have been identified from first principles(conservation of mass, energy, momentum), other dynamic systems have no known equations to characterize their nature. Understanding the characteristics of these systems is vital for developing accurate models for prediction and to inform actuation. 

The challange of determining nonlinearities and associated model parameters from data presents a problem as there are combinatorially many possible model strucutres that can equally describe a dataset. Thus, there exists numerous machine learning algorithms that will yield accurate models that can be used for future state prediction.; however, they do not yield generalizable and interpretable models. This diminishes the ability to characterize a dynamic system of interest and thus limiting the model to be extrapolated to other similar systems.

An interesting observation of dynamic systems is that they only have a few active terms in function space. This has an important implication when attempting to identify governing equations for dynamic systems using data. Sparisty-promoting methods can be used to identifty the few active functions that describe a system. 

The sparse identification of nonlinear dynamics (SINDy) algorithm bypasses the brute-force search through all possible model structures by using sparse regression that penalizes the number of active terms in function space. To demonstrate the SINDy algortihm, consider dynamic systems in the form of

$$
\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t)
$$

where the vector $$\mathbf{x}(t) \in \mathbb{R}^n$$ denotes the state of the system at time $$t$$ and $$\mathbf{f}$$ represents the equations of motion of the system. To determine $$\mathbf{f}$$, time-series data is arranged into a data matrix and a similar matrix of derivatives is also created. Typically, the derivative must be approximated from the data unless it is measured directly from the system. These data matrices are denoted as

$$
\mathbf{X} = \left[\begin{array}{c} \mathbf{x}^{T}(t_{1}) \\ \mathbf{x}^{T}(t_{2}) \\ \vdots \\ \mathbf{x}^{T}(t_{m})\end{array} \right] =  \left[\begin{array}{c} x_{1}(t_{1}) & x_{2}(t_{1}) & \cdots & x_{n}(t_{1}) \\ x_{1}(t_{2}) & x_{2}(t_{2}) & \cdots & x_{n}(t_{2}) \\ \vdots & \vdots & \ddots & \vdots\\ x_{1}(t_{m}) & x_{2}(t_{m}) & \cdots & x_{n}(t_{m})\end{array} \right]
$$


$$
\frac{d}{dt}\mathbf{X} = \left[\begin{array}{c} \frac{d}{dt}\mathbf{x}^{T}(t_{1}) \\ \frac{d}{dt}\mathbf{x}^{T}(t_{2}) \\ \vdots \\ \frac{d}{dt}\mathbf{x}^{T}(t_{m})\end{array} \right] =  \left[\begin{array}{c} \frac{d}{dt}x_{1}(t_{1}) & \frac{d}{dt}x_{2}(t_{1}) & \cdots & \frac{d}{dt}x_{n}(t_{1}) \\ \frac{d}{dt}x_{1}(t_{2}) & \frac{d}{dt}x_{2}(t_{2}) & \cdots & \frac{d}{dt}x_{n}(t_{2}) \\ \vdots & \vdots & \ddots & \vdots\\ \frac{d}{dt}x_{1}(t_{m}) & \frac{d}{dt}x_{2}(t_{m}) & \cdots & \frac{d}{dt}x_{n}(t_{m})\end{array} \right]
$$

The next step is then to construct an augmented matrix which contains the candidate nonlinear functions of the columns of the matrix $$\mathbf{X}$$. This matrix is defined as

$$
\mathbf{\Theta}(\mathbf{X}) = \left[\begin{array}{c} \mathbf{1} & \mathbf{X} & \mathbf{X}^{2} & \cdots & \mathbf{X}^{d} & \cdots  & \sin(\mathbf{X}) & \cos(\mathbf{X}) & \cdots\end{array} \right]
$$

For clarification, each entry in the augmented matrix $$\mathbf{\Theta}(\mathbf{X})$$ is a matrix that has the indicated operation applied to the data matrix $$\mathbf{X}$$. Thus for a matrix that is transformed by the polynomial of degree $$2$$, the resulting matrix is

$$
\mathbf{X}^2 = \left[\begin{array}{c} x^{2}_1(t_{1}) & x_{1}(t_{1})x_{2}(t_{1}) & x_{1}(t_{1})x_{3}(t_{1}) & \cdots & x^{2}_2(t_{1}) & x_{2}(t_{1})x_{3}(t_{1}) & \cdots & x^{2}_n(t_{1}) \\ x^{2}_1(t_{2}) & x_{1}(t_{2})x_{2}(t_{2}) & x_{1}(t_{2})x_{3}(t_{2}) & \cdots & x^{2}_2(t_{2}) & x_{2}(t_{2})x_{3}(t_{2}) & \cdots & x^{2}_n(t_{2}) \\ \vdots & \vdots & \vdots &\ddots & \vdots & \vdots & \ddots & \vdots \\ x^{2}_1(t_{m}) & x_{1}(t_{m})x_{2}(t_{m}) & x_{1}(t_{m})x_{3}(t_{m}) & \cdots & x^{2}_2(t_{m}) & x_{2}(t_{m})x_{3}(t_{m}) & \cdots & x^{2}_n(t_{m}) \end{array}\right]
$$

The augmented matrix $$\mathbf{\Theta}(\mathbf{X})$$ is not limited to polynomial or trigonometric transformations. This matrix can include any function that is believed to describe the system of interest; however, it is important to limit the number of candidate functions in this matrix as it grows extremly fast and thus will scale poorly with large data matrices $$\mathbf{X}$$. The figure below displays the total number of polynomial terms in the agumented matrix at varying polynomial degrees with increasing number of dynamic states (y-axis is displayed on a logarithmic axis for better comparison of the data).

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/SINDy/Polynomial Terms.html" height="525" width="100%"></iframe>

 If the data matrix $$\mathbf{X}$$ contains $$9$$ columns and the system is believed to contain polynomials up to the fifth order, the agumented matrix will contain $$2,002$$ columns (provided that only polynoial terms are included)! 

 Now that the augmented and data matrices have been defined, a sparse regression problem can be constructed in order to identify the functions that can be used to describe the dynamic system. The sparse regression is defined as

$$
\frac{d}{dt}\mathbf{X} = \mathbf{\Theta}(\mathbf{X})\mathbf{\Xi}
$$

The matrix $$\mathbf{\Xi}$$ contains columns that represent a sparse vector of coefficients. This matrix is the objective of the SINDy algorithm and can be determined by utilizing a convex $$L_1$$-regularized sparse regression 

$$
\mathbf{\xi_{k}} = argmin_{\xi^{'}_{k}} \begin{Vmatrix}\frac{d}{dt}\mathbf{X}_{k} - \mathbf{\Theta}(\mathbf{X})\xi^{'}_{k}\end{Vmatrix}_{2} + \lambda\begin{Vmatrix} \xi^{'}_{k}\end{Vmatrix}_{1}
$$

In this expression, $$\frac{d}{dt}\mathbf{X}_{k}$$ is the *k*-th column of $$\frac{d}{dt}\mathbf{X}$$ and $$\lambda$$ controls the manner in which sparsity is promoted for the model. Thus, if $$\lambda = 0$$, the least-squares fit is returned for each vector $$\mathbf{\xi_{k}}$$ and as $$\lambda$$ increases, the degree of sparsity increases

Once $$\mathbf{\Xi}$$ is determined, the governing equations can be constructed by 

$$
\frac{d}{dt}\mathbf{x}_k =\mathbf{\Theta}\mathbf(x)\mathbf{\xi}k
$$

In this expression, $$\mathbf{\Theta}(\mathbf{x})$$ is not the augmented matrix $$\mathbf{\Theta}(\mathbf{X})$$. $$\mathbf{\Theta}(\mathbf{x})$$ represents a row vector of symbolic functions. 

The simple framework of SINDy has made it an increasingly popular method to identify the underlying functions that a dynamic system of interest. To demonstrate this framework in action, a simplified example is presented on a known dynamic system.

## Lorenz System Application

The SINDy algorithm will be applied to the Lorenz system and is defined as

$$
\frac{dx}{dt} = \sigma (y - x)
$$

$$
\frac{dy}{dt} = x (\rho - z) - y
$$

$$
\frac{dz}{dt} = xy - \beta z
$$

The Lorenz system is known to have chaotic solutions for specific parameter values and intial conditions. When these parameters and intial conditions are set properly, the solutions will converge onto the Lorenz attractor. The parameters  $$\sigma=10$$, $$\rho=28$$, and $$\beta=8/3$$ will be used to generate the data. The necessary libraries are first imported

```python
import numpy as np
from scipy.integrate import odeint
from itertools import combinations_with_replacement
from math import factorial
```

Next, the Lorenz system is defined 

```python
def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return x_dot, y_dot, z_dot
```

The data is generated using the lorenz function 

```python
dt = 0.002
t_ = np.arange(0, 10, dt)
x_0 = np.array([-8, 8, 27]) #initial conditions
states = odeint(lorenz, x_0, t_) 
```

The last line in the code block solves the system of differential equations to obtain the spatial coordinates. If we plot the columns of the ```states``` variable, the Lorenz attractor can be visualized

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/SINDy/Lorenz Attractor.html" height="525" width="100%"></iframe>

The ```states``` variable is the data matrix $$\mathbf{X}$$ and if displayed, it should look like

```python
array([[-8.   ,  8.   , 27.   ],
       [-7.684,  7.966, 26.732],
       [-7.374,  7.929, 26.47 ],
       ...,
       [ 7.938, 11.856, 19.626],
       [ 8.017, 11.965, 19.711],
       [ 8.096, 12.074, 19.799]])
```

With respect to the notation used in the overview section, 

$$
\mathbf{X} =  \left[\begin{array}{c} x_{1}(t_{1}) & x_{2}(t_{1}) & \cdots & x_{n}(t_{1}) \\ x_{1}(t_{2}) & x_{2}(t_{2}) & \cdots & x_{n}(t_{2}) \\ \vdots & \vdots & \ddots & \vdots\\ x_{1}(t_{m}) & x_{2}(t_{m}) & \cdots & x_{n}(t_{m})\end{array} \right]
$$

becomes

$$
\mathbf{X} =  \left[\begin{array}{c} -8 & 8 & 27 \\ -7.684 & 7.966 & 26.732 \\ \vdots & \vdots & \vdots\\ 8.096 & 12.074 & 19.799\end{array} \right]
$$

where $$n=3$$

The matrix of derivatives must now be computed. Typically, one will not have this data available and thus must be estimated from the data matrix $$\mathbf{X}$$. One must consider the quality of the data as noisy measurement data causes large variation in the derivative approximation and thus will impact the results of the SINDy algorithm. Different methods for calculating the derivative must be considered to minimize the ill-effects of noisy measurement data.

Because the lorenz function directly calculates the derivative, the topic of derivative estimation is not discussed further in this overview of SINDy. The derivatives can then be estimated using the lorenz function. The following function utilizes the lorenz function as an argument to determine the matrix of derivatives.

```python
def lorenz_derivative(states, lorenz, dt=1):
    x_dot, y_dot, z_dot = lorenz(states[0, :], dt)
    dx_dt = np.array((x_dot, y_dot, z_dot))
    for i in range(1, states.shape[0]):
        x_dot, y_dot, z_dot = lorenz(states[i, :], dt)
        dx_dt = np.vstack((dx_dt, np.array((x_dot, y_dot, z_dot))))
    return dx_dt
```

This function yields the following derivatives matrix

```python
array([[ 160.   ,  -16.   , -136.   ],
       [ 156.498,  -17.713, -132.493],
       [ 153.033,  -19.212, -129.057],
       ...,
       [  39.178,   54.622,   41.785],
       [  39.483,   54.489,   43.365],
       [  39.779,   54.322,   44.959]])
```
and can be visualized by plotting this matrix

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/SINDy/Lorenz Derivative.html" height="525" width="100%"></iframe>

Now that the data matrices $$\mathbf{X}$$ and $$\frac{d}{dt}\mathbf{X}$$ are defined, the SINDy algorithm can be implemented. The augmented matrix is computed with the following functions

```python
def polynomial_combination(states, degrees):
    combinations = factorial(degrees + states) / (factorial(degrees) * factorial(states))
    return int(combinations)

def Theta_library(X, poly_power):
    states = X.shape[1] #states
    m = X.shape[0] #total number of times in data matrix
    poly_space = polynomial_combination(states, degrees=poly_power)
    Theta = np.ones((m, poly_space)) # m x p matrix of ones
    counter = 1 #refernce columns of theta
    column_idx = list(range(states)) #reference columns of data matrix X
    for i in range(poly_power+1):
        for combo in combinations_with_replacement(column_idx, i):
            temp_ones= np.ones((m, 1))
            for c in combo:
                temp_ones = temp_ones * X[:, c].reshape(m, 1)
            Theta[:, (counter - 1):counter] = temp_ones
            counter += 1
    return Theta    
```

The function ```polynomial_combination``` determines the total number of linear and nonlinear functions that can be constructed given the number of states of the data and the desired orders of polynomials. Mathematically, this function calculates

$$
\frac{(degrees + states)!}{degrees!\times states!}
$$

Thus for a system of $$3$$ states ($$x, y, z$$) if 2nd order polynomials are desired, a total $$10$$ combinations of the $$3$$ states can be generated and they are

$$
1, x, y, z, x^{2}, xy, xz, y^{2}, yz, z^{2}
$$

and in matrix notation

$$
\mathbf{\Theta}(\mathbf{X}) = \left[\begin{array}{c} \mathbf{1} & \mathbf{X} & \mathbf{X}^{2} \end{array} \right]
$$

where 

$$
\mathbf{1} = \left[\begin{array}{c}\vert \\1\\ \vert\end{array} \right] \space \space \mathbf{X} =\left[\begin{array}{c}\vert & \vert &\vert \\ \mathbf{x} & \mathbf{y} & \mathbf{z} \\ \vert & \vert &\vert \end{array} \right] \space \space \mathbf{X^{2}} = \left[\begin{array}{c}\vert & \vert &\vert & \vert &\vert&\vert\\ \mathbf{x}^{2} & \mathbf{xy} & \mathbf{xz} & \mathbf{y}^{2} & \mathbf{yz} & \mathbf{z}^{2} \\\vert & \vert &\vert&\vert & \vert &\vert\end{array} \right]
$$

```Theta_library``` uses the calculation of ```polynomial_combination``` to create a matrix of correct dimensions. The funciton then carries out a series of loops that carry out the necessary multiplications and sequentally build the augmented matrix.

This function can be tested on a small matrix to ensure it is working as expected. The test matrix defined as

$$
\mathbf{X} =\left[\begin{array}{c}1 & 2 & 3\\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{array} \right]
$$

thus if 2nd order polynomials are to be calculated, it is expected that the resulting augmented matrix will be

$$
\mathbf{X} =\left[\begin{array}{c}1 & 1 & 2 & 3 & 1 & 2 & 3 & 4 & 6 & 9 \\ 1 & 4 & 5 & 6 & 16 & 20 & 24 & 25 & 30 & 36 \\ 1 & 7 & 8 & 9 & 49 & 56 & 63 & 64 & 72 & 81 \end{array} \right]
$$

Executing the following code block

```python
Theta_x = Theta_library(X, 2)
print(Theta_x)

[[ 1.  1.  2.  3.  1.  2.  3.  4.  6.  9.]
 [ 1.  4.  5.  6. 16. 20. 24. 25. 30. 36.]
 [ 1.  7.  8.  9. 49. 56. 63. 64. 72. 81.]]
```
the expected results are verified. The augmented matrix is constructed using the generated data with 3rd degree polynomials.

```python
Theta_x = Theta_library(states, 3)
```

The sparse regression can then be computed. In this example, the sequential thresholded least-squares (STLS) algorithm is utilized to determine the matrix $$\mathbf{\Xi}$$ although other sparsity-promoting regression techniques like LASSO could be utilized instead. The algorithm is defined as

```python
def sparsify_dynamics(Theta, dx_dt, n=10, lambda_=0.025):
    Xi = np.linalg.lstsq(Theta, dx_dt, rcond=None)[0]
    for i in range(n):
        Xi = np.where(np.abs(Xi) < lambda_, 0, Xi)
        for j in range(Xi.shape[1]):
            mask = np.abs(Xi[:, j]) > lambda_
            Xi[:, j][mask] = np.linalg.lstsq(Theta[:, mask], dx_dt[:, j], rcond=None)[0]
    return Xi
```

The algorithm begins by determining the least-squares fit for the matrix $$\mathbf{\Xi}$$. All coefficients that are less than $$\lambda$$ are set to $$0$$. Each column vector in the matrix $$\mathbf{\Xi}$$ whose entries are non-zero are used to obtain another least-squares solution to $$\mathbf{\Xi}$$. This process is repeated until the non-zero coefficients converge.

The STLS is applied using the augmented matrix and the derivative matrix as the two required arguments.

```python
Xi = sparsify_dynamics(Theta_x, dx_dt)
print(Xi)

[[  0.      0.      0.   ]
 [-10.     28.      0.   ]
 [ 10.     -1.      0.   ]
 [  0.      0.     -2.667]
 [  0.      0.      0.   ]
 [  0.      0.      1.   ]
 [  0.     -1.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]
 [  0.      0.      0.   ]]
```

Notice that the coefficients correspond to the parameters used for the lorenz system $$\sigma=10$$, $$\rho=28$$, and $$\beta=8/3$$. The results can be interpreted easier if we add the row vector of symbolic functions $$\mathbf{\Theta}(\mathbf{x})^{T}$$ to $$\mathbf{\Xi}$$. 

```python
[['1' '0.0' '0.0' '0.0']
 ['x' '-10.00' '28.000' '0.0']
 ['y' '10.0' '-0.999' '0.0']
 ['z' '0.0' '0.0' '-2.666']
 ['xx' '0.0' '0.0' '0.0']
 ['xy' '0.0' '0.0' '0.9999']
 ['xz' '0.0' '-1.000' '0.0']
 ['yy' '0.0' '0.0' '0.0']
 ['yz' '0.0' '0.0' '0.0']
 ['zz' '0.0' '0.0' '0.0']
 ['xxx' '0.0' '0.0' '0.0']
 ['xxy' '0.0' '0.0' '0.0']
 ['xxz' '0.0' '0.0' '0.0']
 ['xyy' '0.0' '0.0' '0.0']
 ['xyz' '0.0' '0.0' '0.0']
 ['xzz' '0.0' '0.0' '0.0']
 ['yyy' '0.0' '0.0' '0.0']
 ['yyz' '0.0' '0.0' '0.0']
 ['yzz' '0.0' '0.0' '0.0']
 ['zzz' '0.0' '0.0' '0.0']]
```

Column $$2$$ corresponds to $$\frac{dx}{dt} = 10 (y - x)$$, column $$3$$ to $$\frac{dy}{dt} = x (28 - z) - y$$, and column $$4$$ to $$\frac{dz}{dt} = xy - 8/3 z$$. 

The expression 

$$
\frac{d}{dt}\mathbf{x}_k =\mathbf{\Theta}(\mathbf{x})\mathbf{\xi}k
$$ 

can then be evaluated as follows

$$
\frac{d}{dt}\mathbf{x}_1 = \frac{d}{dt}\mathbf{x}= \left[\begin{array}{c}1 & x & y & z & xx & xy & xz & yy & yz & zz & \cdots & zzz \\  \end{array} \right]\left[\begin{array}{c}0 \\ -10 \\ 10 \\ 0 \\ 0\\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ \vdots \\ 0 \\  \end{array} \right] = -10x + 10y
$$

$$
\frac{d}{dt}\mathbf{x}_2 = \frac{d}{dt}\mathbf{y}= \left[\begin{array}{c}1 & x & y & z & xx & xy & xz & yy & yz & zz & \cdots & zzz \\  \end{array} \right]\left[\begin{array}{c}0 \\ 28 \\ -0.999 \\ 0 \\ 0\\ 0 \\ -1 \\ 0 \\ 0 \\ 0 \\ \vdots \\ 0 \\  \end{array} \right] = -28x - 0.999y -xy 
$$

$$
\frac{d}{dt}\mathbf{x}_3 = \frac{d}{dt}\mathbf{z}= \left[\begin{array}{c}1 & x & y & z & xx & xy & xz & yy & yz & zz & \cdots & zzz \\  \end{array} \right]\left[\begin{array}{c}0 \\ 0 \\ 0 \\ -2.667 \\ 0\\ 0.999 \\ 0 \\ 0 \\ 0 \\ 0 \\ \vdots \\ 0 \\  \end{array} \right] = -2.667z + 0.999xy
$$

Although the algorithm was demonstrated as a series of sequential functions, in practice, it is more practical to design a python class with the nescessary methods to perfom the sequence of operations that will output the desired results. To demonstrate this idea, a class was designed to generate the final output. This python class is defined as

```python
class SINDy(object):
    """SINDy class to determine active terms in function space of a 
    given dataset of a dynamic system.
    """
    
    def __init__(self, lambda_=0.025, n=10, poly_power=3):
        self.lambda_ = lambda_
        self.n = n
        self.poly_power = poly_power
        self._m = None
        self._Theta = None
        self._Xi = None
        self._functions = []
        self._feature_names = ["x'", "y'", "z'"]
        
    def fit(self, X, dx_dt):
        self._Xi = self.sparsify_dynamics(X, dx_dt)
        self._functions = self.function_vector()
        Xi = np.column_stack((self._functions, self._Xi.astype(dtype=np.dtype("<U6"))))
        return Xi
        
    def polynomial_combination(self, states, degrees):
        combinations = factorial(degrees + states) / (factorial(degrees) * factorial(states))
        return int(combinations)
    
    def Theta_library(self, X):
        self._states = X.shape[1]
        self._m = X.shape[0]
        poly_space = self.polynomial_combination(self._states, degrees=self.poly_power)
        self._Theta = np.ones((self._m, poly_space))
        counter = 1
        column_idx = list(range(self._states))
        for i in range(self.poly_power+1):
            for combo in combinations_with_replacement(column_idx, i):
                temp_ones= np.ones((self._m, 1))
                for c in combo:
                    temp_ones = temp_ones * X[:, c].reshape(self._m, 1)
                self._Theta[:, (counter - 1):counter] = temp_ones
                counter += 1
        return self._Theta
    
    def sparsify_dynamics(self, X, dx_dt,):
        self._Theta = self.Theta_library(X)
        Xi = np.linalg.lstsq(self._Theta, dx_dt, rcond=None)[0]
        for i in range(self.n):
            Xi = np.where(np.abs(Xi) < self.lambda_, 0, Xi)
            for j in range(Xi.shape[1]):
                mask = np.abs(Xi[:, j]) > self.lambda_
                Xi[:, j][mask] = np.linalg.lstsq(self._Theta[:, mask], dx_dt[:, j], rcond=None)[0]
        return Xi
    
    def function_vector(self):
        for i in range(self.poly_power+1):
            if i == 0:
                self._functions.append("1")
            else:
                for combo in combinations_with_replacement("xyz", i):
                    self._functions.append("".join(combo))
        return np.array(self._functions)
    
    def equations(self):
        eqn_system = []
        for i in range(self._states):
            eqn = self._functions[np.argwhere(np.abs(self._Xi[:, i]) > 0)].flatten()
            coeff = self._Xi[:, i][np.argwhere(np.abs(self._Xi[:, i]) > 0)].flatten()
            coeff = coeff.astype(dtype="<U6")
            p=self._feature_names[i] + " = "
            for j in range(len(eqn)):
                if j != list(range(len(eqn)))[-1]:
                    p = p +  str(coeff[j] + eqn[j] + " + ")
                else:
                    p = p + str(coeff[j] + eqn[j])
            eqn_system.append(p)
            
        return eqn_system
```

Once a class with the nescessary methods and attributes has been defined, an instance of this class can be created and ran on the data as follows

```python
sindy = SINDy()
sindy.fit(states, dx_dt)
sindy.equations()

["x' = -10.00x + 10.0y",
 "y' = 28.000x + -0.999y + -1.000xz",
 "z' = -2.666z + 0.9999xy"]
```

This methodolgy offers a simple and concise manner of reaching the desired output of the SINDy algorith.

Despite no information of the underlying equations being introduced into the algorithm, SINDy manages to identify the correct sparse vectors of coefficients with minescule error. This opens big opportunites to study systems with unknown equations and observe what results can be drawn from the SINDy algortihm. 