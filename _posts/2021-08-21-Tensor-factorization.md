---
layout: post
title: "Tensor factorization"
author: "Lam"
categories: journal
tags: [documentation,sample]
---





> Bayesian Poisson Tensor Factorization for Inferring Multilateral Relations from Sparse Dyadic Event Counts

[2015 Pointer](https://arxiv.org/pdf/1506.03493.pdf)

#### Bayesian Poisson Tensor Factorization

Gvien an $M$-way tensor $Y$, tensor factorization decomposes $Y$ into $M$ latent factor matrices $\Theta^{(1)}, \Theta^{(2)}, \ldots \Theta^{(M)}$ . Two common factorization methods are the [Tucker decomposition](https://en.wikipedia.org/wiki/Tucker_decomposition) and the [Canonical Polyadic (CP) decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition).



For sparse count data, it is recommended to consider the CP decomposition ([reference](https://ieeexplore.ieee.org/document/4781131)).



Given the count tensor $Y \in \mathbb{N}^{N \times T \times R}$ (size #neurons $\times$ #time bin $\times$ #trials), the CP decomposition represents each count $y_{itr}$ as


$$
y_{itr} \approx \hat{y}_{itr} = \sum^K_{k=1} \theta^{(1)}_{ik} \theta^{(2)}_{tk}\theta^{(3)}_{rk}, \quad i \in [N],\ t \in [T],\ r \in [R]
$$


where $\hat{y}_{itr}$ is usually called **reconstruction count**. By aggregrating factors, we have


$$
\Theta = \left\{
\begin{align}
&\Theta^{(1)} = \left[ (\theta^{(1)}_{ik})_{i=1}^N\right]_{k=1}^K \in \mathbb{R}^{N \times K},
\\
&\Theta^{(2)} = \left[ (\theta^{(1)}_{tk})_{t=1}^T \right]_{k=1}^K \in \mathbb{R}^{T \times K},
\\
&\Theta^{(3)} = \left[ (\theta^{(1)}_{rk})_{r=1}^R \right]_{k=1}^K \in \mathbb{R}^{R \times K}
\end{align}
\right\}
$$


By considering $\hat{y}_{itr}$ as the mean of a Poisson distribution (modeling the observed count), i.e., $y_{itr} \sim \text{Poisson}(\hat{y}_{itr})$, the decomposition is known as **Poisson tensor factorization**.



To impose prior distributions on the latent factors, full Bayesian inference is performed. For computational efficiency, conjugate prior is considered. Since the Gamma distribution is the conjugate prior for Poisson likelihood, gamma priors on latent factors are imposed.



Recall that if $\theta \sim \Gamma(a, b)$, then $\mathbb{E}[\theta] = \frac{a}{b}$ and $\text{Var}[\theta] = \frac{a}{b^2}$. If $a \ll 1$ and $b$ is small, the distribution concentrates mostly near zero, yet a heavy tail is still maintained. Therefore it can be used as a sparsity-inducing prior (comparison with Laplace prior ?)



For every single factor, e.g. $\theta^{(1)}_{ik}$, a sparsity gamma priors is imposed:


$$
\theta^{(1)}_{ik} \sim \Gamma( \alpha, \alpha\beta^{(1)})
$$


where the rate parameter is the product of the shape parameter and $\beta^{(1)}$. It follows that the mean of the prior is completely determined by $\beta^{(1)}$ ($\mathbb{E}(\theta^{1}_{ik}) = \frac{1}{\beta^{(1)}}$). The shape parameter, which determins the sparsity of the factor matrices, could be set using data driven approach (e.g. cross validation scheme).



#### Posterior estimate using variational inference

Let $\mathcal{H} = \{ \alpha, \beta^{(1)}, \beta^{(2)}, \beta^{(3)}\}$ to be the set of prior parameters. The posterior is approximated by a parametric distribution $Q$. The authors in this paper choose a fully factorized mean-field approximation. Specifically,


$$
p(\Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}  \ \vert \ Y, \mathcal{H}) \approx Q, \quad \text{where}
\\
Q = \prod^K_{k=1} \prod^N_{i=1} \Gamma(\gamma_{ik}, \delta_{ik}) \prod^T_{t=1} \Gamma(\gamma_{tk}, \delta_{tk}) \prod^R_{r=1} \Gamma(\gamma_{rk}, \delta_{rk}) 
$$


Therefore the full set of variational parameters is


$$
 \mathcal{S} = \{ ((\gamma_{ik}, \delta_{ik})^N_{i=1})_{K=1}^k, \quad ((\gamma_{ik}, \delta_{tk})^T_{t=1})_{K=1}^k, \quad ((\gamma_{rk}, \delta_{rk})^R_{r=1})_{K=1}^k\}
$$


In VI, $\mathcal{S}$ is set to those that minimize the KL divergence with the exact posterior:


$$
\begin{align}
& \quad D_{KL} ( Q(\mathcal{S})  \ \Vert \ p(\Theta \vert Y, \mathcal{H}))
\\
&= - H(Q) - \mathbb{E}_Q \left[ \log \frac{p(Y \vert \mathcal{H}, \mathcal{S}) p(\mathcal{S})}{p(Y \vert \mathcal{H})} \right]
\\
&= - H(Q) - \mathbb{E}_Q\left[ \log p(Y \vert \mathcal{H}, \mathcal{S} ) p ( \mathcal{S} )\right] + \text{constant}
\\
&= \underbrace{- H(Q) - \mathbb{E}_Q[ \log p(Y, \mathcal{S} \ \vert \ \mathcal{H})]}_{ELBO} + \text{constant}
\end{align}
$$
We can minimize the KL divergence by maximizing the $ELBO$. Maximizing $ELBO$ can be achieved by coordinate ascent method.







 