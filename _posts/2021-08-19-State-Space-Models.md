---
layout: post
title: "State Space Model"
author: "Lam"
categories: journal
tags: [documentation,sample]
---



# State Space Model

### Intuition

General model for sequential data:
$$
p(y_1, y_2, \ldots, y_T) = \prod^T_{t=1} p(y_t | y_1, \ldots y_{t-1})
$$
Easiest way to treat sequential data would be the (first order) **Markov model**:
$$
p(y_1, y_2, \ldots, y_T) = p(y_1) \cdot \prod^T_{t=2} p(y_t | y_{t-1})
$$
However, it is restrictive since it is equivalent to assume $p(y_t |y_1,\ldots y_{t-1}) = p(y_t | y_{t-1})$. Generalize to second order, we have
$$
p(y_1, y_2, \ldots, y_T) = p(y_1)p(y_2|y_1) \cdot \prod^T_{t=3} p(y_t | y_{t-1}, y_{t-2})
$$
at the expense of increasing number of model parameters which grow exponentially with the markov chain order.

To build a model that is **not limited by Markov assumption to any order** (but with limited parameters), latent variables are introduced. For each ovbservation $y_t$, we introduce $x_t$ and assume the latent variables form a Markov chain (but not $y_t$). The joint distribution of this model is
$$
p(y_1, y_2, \ldots y_T, x_1, x_2, \ldots, x_T) = p(x_1) \left[\prod^T_{t=2} p(x_t | x_{t-1}) \right] \prod^T_{t=1} p(y_t|x_t)
$$
Latent variables provide path connecting any $y_t$ and $y_{t'}$. Thus $y_t$ depends on all previous observation.







### Poisson LDS model

$y_t = [y_{1,t}, \ y_{2, t}, \ \ldots, \ y_{N, t}]$:  vector representing activity of $N$ neurons at time $t$ (e.g. spike count / calcium fluorescence)

$x_t$ : continuous latent state at $t$, $x_t \in \mathbb{R}^D$, $D \ll N$

$z_t = [z_{1,t}, \ z_{2, t}, \ \ldots, z_{N, t}]$: intermediate variables defined as
$$
z_t = C x_t + D\rho_t + d
$$
where $\rho_t$ is some external covariates. For example, one choice would be $\rho_t = y_{t-1}$.

We assume that, given $z_{i,t}$, the spike count of neuron $i$ in bin $t$ is sample from a Poisson distribution with mean $\eta(z_{i,t})$, for some nonlinear, non-negative function (e.g. $\eta(\cdot) = \exp(\cdot)$ or $\eta(\cdot) = \log ( 1 + \exp(\cdot)))$. Then
$$
p(y_{i,t} \ | \ z_{i,t}) = p(y_{i,t} \ | \ [Cx_t + D y_{t-1} + d]_i) = \frac{1}{y_{i,t}!} \eta(z_{i,t})^{y_{i,t}} \exp(-\eta(z_{i,t}))
$$

> Sometimes $z_{i,t}$ is called **pre-intensity**, where $\eta(z_{i,t})$ would be the **intensity** of Poisson point process. In particular, consider the Poisson point process, which is a stochastic process over a bounded domain $S$ (i.e. the time bin $t$ of neuron $i$). The "points" are scattered over $S$ and the number of "points" (i.e.,  spike count), denoted by $n(S)$, has a Poisson distribution.

​           

The $N \times D$ loading matrix $C$ is sometimes called the **emission matrix** since $C$ determins how each neuron is influenced by the latent state $x_t$. 



Lastly, the state $x_t$ dyanmics could be described by linear Guassian
$$
x_1 = x_0 + \epsilon_0, \quad \epsilon_0 \sim \mathcal{N}(0, Q_0)
\\
x_t = Ax_{t-1} + Bu_t + \epsilon, \quad \epsilon \sim \mathcal{N}(0, Q)
$$
Different choice of $u_t$ could be considered. For example, $u_t$ is an indicator function of current time $t$; or $u_t = \mathbb{1}$ to turn $B$ into a simple bias.



### SLDS model

Extending LDS to Switching LDS is straightforward. Instead of having one homogeneous state for the whole time domain, we consider $K$ states, denoted as $s_t \in \{1, 2, \ldots, K\}$. Each state has its own dynamics and intensity:
$$
x_t = A^{(s_t)}x_{t-1} + B^{(s_t)} u_t + \epsilon^{(s_t)}, \quad \epsilon^{(s_t)} \sim \mathcal{N}(0, Q^{(s_t)})
\\
\eta(z_{i,t}) = \eta( C^{(s_t)} x_{t} + D^{(s_t)}\rho_t + d^{(s_t)})
$$
The state transition probabilities, $p(s_t |s_{t-1})$, are specified by a Markov chain (i.e., a $K \times K$ transition matrix).

### rSLDS model

The discrete states $s_t$ in rSLDS model are modeled as a Markov process using categorical distribution
$$
s_t \sim \text{Cat}(\pi_t), \quad \pi_t = \text{softmax}(R^{(s_t)}x_{t-1} + r^{(s_t)}), \quad R^{(s_t)} \in \mathbb{R}^{K \times D}, r^{(s_t)} \in \mathbb{R}^K
$$


### Optimization

Two main optimization approaches are `Laplace-EM` algorithm and the `Variational Inference` algorithm.

`Laplace-EM` algorithm usually perform better and is more efficient. However, it is more restrictive (Gaussian dynamics).

`Variational Inference` is general but empirical results may not be as good as `Laplace-EM`.

#### Poisson LDS

For simplicity, we assume $\eta(\cdot) = \exp(\cdot)$ and now external covariates are present ($D = B = 0$). Recall the model
$$
z_t = C x_t + d
\\
x_1 = x_0 + \epsilon_0, \quad \epsilon_0 \sim \mathcal{N}(0, Q_0)
\\
x_t = Ax_{t-1} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, Q)
\\
p(y_{i,t} \ | \ z_{i,t}) = \frac{1}{y_{i,t}!} \eta(z_{i,t})^{y_{i,t}} \exp(-\eta(z_{i,t}))
$$
The posterior distribution $p(x_{1:T} | y_{1:T})$ does not correspond to a well-studied standard distribution with closed form expectation. Therefore, we approximate $p(x_{1:T} | y_{1:T})$ with an appropriate distribution $q(x_{1:T})$.

Simplify the notation $x_{1:T}$ and $y_{1:T}$ by $\mathbf{x} = \text{vec}(x_{1:T})$ and $ \mathbf{y} = \text{vec}(y_{1:T})$. By choosing $\eta(\cdot) = \exp(\cdot)$, the posterior $P(\mathbf{x}| \mathbf{y})$ has a single peak (?) and so we consider $q(\mathbf{x}) = q(\mathbf{x} | \mu, \Sigma) = \mathcal{N}(\mu, \Sigma)$, with normal density function
$$
\phi(\mathbf{x} | \mu, \Sigma) = (2\pi)^{-pT/2} |\Sigma|^{-1/2} \exp\left( -\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu) \right)
$$
**Optimize for $\mu$ and $\Sigma$ using Laplace approximation**

Notice that the log density is highest at the unique point $x = \mu$. Conversely, we have
$$
\mu = \arg\max_{\mathbf{x}} \log \phi(\mathbf{x} | \mu, \Sigma)
$$
Hessian of the log density also yields the inverse covariance matrix
$$
\Sigma^{-1} = -\nabla^2_x \log \phi( \mathbf{x} | \mu, \Sigma)
$$
Therefore, our procedure is

- find the log-posterior over $\mathbf{x}$
- find its optimal (using gradient, for instance)
- find its Hessian

By Bayes's rule, the log-posterior is
$$
\log p(\mathbf{x} | \mathbf{y}) = \log p(\mathbf{y} | \mathbf{x}) + \log p(\mathbf{x}) + \text{Constant}(\mathbf{x})
$$
where the log-likelihood is
$$
\begin{split}
\log p(\mathbf{y} | \mathbf{x}) &= \sum^T_{t=1} \sum^N_{i=1} \log p(y_{i,t}| \mathbf{x})
\\
&=\sum^T_{t=1} \sum^N_{i=1} \log \left(  \frac{1}{y_{i,t}!} \eta(z_{i,t})^{y_{i,t}} \exp(-\eta(z_{i,t})) \right)
\\
&= \sum^T_{t=1} \sum^N_{i=1} y_{i,t} z_{i,t} - \exp(z_{i,t}) + \text{Constant}
\\
&= \sum^T_{t=1} \sum^N_{i=1} y_{i,t} \left( (C\mathbf{x_t})_i + d_i \right) - \exp\left( (C\mathbf{x_t})_i + d_i \right) + \text{Constant}
\\
&= \mathbf{y}^T \left( (\mathbb{I}_T \otimes C)\mathbf{x} + \mathbf{1}_T \otimes d \right) - \mathbf{e}_{NT}^T \exp( (\mathbb{I}_T \otimes C)\mathbf{x} + \mathbf{1}_T \otimes d ) + \text{Constant}
\\
&= \mathbf{y}^T \left( \widetilde{C} \mathbf{x} + \widetilde{d} \right) - \mathbf{e}_{NT}^T \exp \left( \widetilde{C} \mathbf{x} + \widetilde{d} \right) + \text{Constant}

\end{split}
$$
The prior over $\mathbf{x}$ is defined by the dynamics
$$
\begin{split}
\log p(\mathbf{x}) &= \log \phi( \mathbf{x}_1 | \mathbf{x}_0, Q_0) + \sum^T_{t=2} \log \phi(\mathbf{x}_1 | \mathbf{x}_{t-1}, Q)
\\
&= -\frac{1}{2} (\mathbf{x}_1 - \mathbf{x}_0)^T Q_0^{-1} (\mathbf{x}_1 - \mathbf{x}_0) \\
& \quad \quad- \frac{1}{2} \sum^T_{t=2} \left( \mathbf{x}_t - A\mathbf{x}_{t-1} \right)^T Q^{-1} \left( \mathbf{x}_t - A\mathbf{x}_{t-1} \right) + \text{Constant}(\mathbf{x})
\end{split}
$$
Note that the joint distribution over $\mathbf{x}_t$ ( or equivalently, distributoon over $\mathbf{x}$) is multivariate normal. In particular, grouping first and second order terms, the prior mean $\mu_{\pi}$ and covariance $\Sigma_{\pi}$ can be expressed using $A, Q, Q_0$ and $x_0$:
$$
\Sigma^{-1}_{\pi} = \begin{bmatrix} Q_0^{-1} + A^T Q^{-1} A & -A^T Q^{-1} & \cdots & 0 \\
-Q^T A & A^T Q^{-1} A + Q^{-1} & -A^T Q^{-1} & \vdots \\
\vdots & -Q^{-1}A & A^T Q^{-1}A + Q^{-1} & -A^T Q^{-1} \\
0 & \cdots & -Q^{-1}A & Q^{-1}\end{bmatrix}
$$

$$
\mu_{\pi} = \begin{bmatrix} x_0 & Ax_0 & \cdots & A^{T-1}x_0 \end{bmatrix}^T
$$

Combining both, the log posterior (neglecting constant terms)
$$
L(\mathbf{x}) := \log p(\mathbf{x} | \mathbf{y}) = \mathbf{y}^T \left( \widetilde{C} \mathbf{x} + \widetilde{d} \right) - \mathbf{e}_{NT}^T \exp \left( \widetilde{C} \mathbf{x} + \widetilde{d} \right) - \frac{1}{2} (\mathbf{x} - \mu_{\pi})^T \Sigma^{-1}_{\pi} (\mathbf{x} - \mu_\pi)
$$
Its gradient and Hessian are:
$$
\nabla_{\mathbf{x}} L = \widetilde{C}^T \mathbf{y} - \widetilde{C}^T \exp\left( \widetilde{C} \mathbf{x} + \widetilde{d} \right) - \Sigma^{-1}_{\pi} (\mathbf{x} - \mu_\pi)
\\
\nabla_{\mathbf{x}}^2 L = -\widetilde{C}^T \text{diag} \left( \left( \widetilde{C} \mathbf{x} + \widetilde{d} \right)\right) \widetilde{C} - \Sigma^{-1}_{\pi}
$$
**Optimize for $\mu$ and $\Sigma$ using Variational inference**

One approximates the posterior with a distribution $q(\mathbf{x})$ according to the KL divergence between them:
$$
\mu^*, \Sigma^* = \arg\min_{\mu, \Sigma} D_{KL}(q(\mathbf{x}) || p(\mathbf{x} | \mathbf{y}))
$$
Recall that 
$$
\begin{split}
D_{KL}( q(\mathbf{x}) \ || \ p(\mathbf{x}|\mathbf{y})) ) &= \mathbb{E}_{q(\mathbf{x})} \left[ \log \frac{q(\mathbf{x})}{p( \mathbf{x}|\mathbf{y} )}\right]
\\
&= \mathbb{E}_{ q(\mathbf{x})} [ \log q(\mathbf{x}) ] - \mathbb{E}_{q(\mathbf{x})} \left[ \log \frac{ p(\mathbf{y}|\mathbf{x}) p(\mathbf{x})}{p(\mathbf{y})}\right]
\\
&=  \mathbb{E}_{ q(\mathbf{x})} [ \log \frac{q(\mathbf{x})}{p(\mathbf{x})} ]  - \mathbb{E}_{q(\mathbf{x})} \left[ \log  p(\mathbf{y}|\mathbf{x}) \right] + \mathbb{E}_{q(\mathbf{x})} \left[ \log  p(\mathbf{y}) \right]
\\
&= \underbrace{D_{KL}( q(\mathbf{x} \ || \ p(\mathbf{x}) )) - \mathbb{E}_{q(\mathbf{x})} \left[ \log  p(\mathbf{y}|\mathbf{x}) \right]}_{\text{Negative Lower bound}}  + \mathbb{E}_{q(\mathbf{x})} \left[ \log  p(\mathbf{y}) \right]
\end{split}
$$
where the KL divergence between two Guassian distribution is ( [reference](https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/) ):
$$
D_{KL}( q(\mathbf{x} \ || \ p(\mathbf{x}) )) = \frac{1}{2} \left[ \log \frac{|\Sigma_{\pi}|}{|\Sigma|} - DT + (\mu - \mu_\pi)^T \Sigma^{-1} (\mu - \mu_\pi)  + \trace(\Sigma^{-1}_\pi \Sigma)\right]
$$
As $\mathbb{E}_{q(\mathbf{x})} \left[ \log  p(\mathbf{y}) \right]$ is constant for any $\mathbf{x}$, minimizing $D_{KL}( q(\mathbf{x}) \ || \ p(\mathbf{x}|\mathbf{y})) )$ is equivalent to maximizing the  following lower bound (up to constant):
$$
\mathcal{L}(\mu, \Sigma) = \frac{1}{2} \left[ \log {|\Sigma|} - (\mu - \mu_\pi)^T \Sigma^{-1} (\mu - \mu_\pi) - \trace(\Sigma^{-1}_\pi \Sigma)\right] + \mathbb{E}_{q(\mathbf{x})} \left[ \log  p(\mathbf{y}|\mathbf{x}) \right]
$$
where
$$
\begin{split}
\mathbb{E}_{q(\mathbf{x})} [ \log p(\mathbf{y} | \mathbf{x})] &= \mathbb{E}_{q(\mathbf{x})} \left[ \mathbf{y}^T \left( \widetilde{C} \mathbf{x} + \widetilde{d} \right) - \mathbf{e}_{NT}^T \exp \left( \widetilde{C} \mathbf{x} + \widetilde{d} \right) \right]
\\
&= \mathbf{y}^T \left( \widetilde{C} \mu + \widetilde{d} \right) - \mathbf{e}_{NT}^T \exp \left( \widetilde{C} \mu + \widetilde{d}  + \frac{1}{2}{\text{diag}}(\widetilde{C}\Sigma \widetilde{C}^T)\right)
\end{split}
$$


> More about the tricks to optimize will be summarized later



**Estimate parameters using Expectation maximization**

