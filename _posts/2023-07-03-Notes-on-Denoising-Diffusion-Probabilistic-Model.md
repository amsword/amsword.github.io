---
layout: post
comments: true
title: Notes on Denoising Diffusion Probabilistic Models
author: Jianfeng Wang
---

## Forward pass
Let $$\mathbf{x}_0$$ be the observed sample and follow an unknown
distribution $$q(\mathbf{x}_0)$$. The forward pass is to disturb the data by
shrinking the mean and adding extra noise. That is


$$
q(\mathbf{x}_{t}|\mathbf{x}_{t - 1}) \sim \mathcal{N}(\sqrt{1 - \beta_t}\mathbf{x}_{t - 1}, \beta_{t}\mathbf{I}), t = 1,\cdots, N.
$$


We can also write $$\mathbf{x}_{t}$$ as follows.

$$
\mathbf{x}_{t} = \sqrt{1 - \beta_{t}} \mathbf{x}_{t - 1} + \sqrt{\beta_{t}} \epsilon_t,
$$

where $\epsilon_t$ follows the normal Gaussian distribution $$\mathcal{N}(\mathbf{0}, \mathbf{I})$$.
Next, let's derive the probability distribution of $\mathbf{x}_{t}$, given $\mathbf{x}_0$.
Similarly, we have

$$
\mathbf{x}_{t-1} = \sqrt{1 - \beta_{t - 1}} \mathbf{x}_{t - 2} + \sqrt{\beta_{t - 1}} \epsilon_{t - 1}   \\
\vdots \\
\mathbf{x}_{1} = \sqrt{1 - \beta_{1}} \mathbf{x}_{0} + \sqrt{\beta_{1}} \epsilon_{1} \\
$$

Each equation is multiplied by a factor such that the multiplied left side is
equal to the one on the right of the previous equation. Thus, we have

$$
\mathbf{x}_{t} = \prod_{i = 1}^{t} \sqrt{1 - \beta_{i}} \mathbf{x}_0 + f(\epsilon_1, \cdots, \epsilon_t)  \\
f(\epsilon_1, \cdots, \epsilon_t) = \sqrt{\beta_{t}} \epsilon_t + \cdots + \prod_{i=2}^{t} \sqrt{1 - \beta_i} \sqrt{\beta_1} \epsilon_1
$$

Considering $\epsilon_i$ are i.i.d, we can simply calculate the mean and
variance of $f$. As the mean of each $\epsilon_i$ is 0, the mean of $f$ is also 0, as the mean of the sum of multiple random variable equals the sum of the mean of each variable.
The variance of $f$ is also the sum of the variance of each variable when each
variable is independently distributed. That is,

$$
\beta_{t} + (1 - \beta_{t}) \beta_{t - 1} + \cdots + \prod_{i=2}^{t} (1 - \beta_i) \beta_1 =
\beta_{t} + \sum_{k = 2}^{t} \prod_{i = k}^{t}(1 - \beta_{i}) \beta_{k - 1}
$$

Replacing $\alpha_i = 1 - \beta_i$, we can rewrite the variance as

$$
1 - \alpha_t + \sum_{k = 2}^{t} \prod_{i = k}^{t} \alpha_{i} (1 - \alpha_{k - 1}) = 
1 - \alpha_t + \sum_{k = 2}^{t} (
\prod_{i = k}^{t} \alpha_{i} - \prod_{i = k - 1}^{t} \alpha_{i}
) = 1 - \prod_{i = 1}^{t} \alpha_{i}.
$$

Thus, we have

$$
q(\mathbf{x}_{t} | \mathbf{x}_{0}) \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_{t})\mathbf{I}) \\
\bar{\alpha}_{t} = \prod_{i = 1}^{t} \alpha_{i}, \alpha_i = 1 - \beta_i
$$

Another way to derive $$q(\mathbf{x}_t | \mathbf{x}_0)$$ is by induction. When
$$t = 1$$, it is straightfward to have the result. Let's say we have
$$q(\mathbf{x}_{t - 1} | \mathbf{x}_0) =
\sqrt{\bar{\alpha}_{t - 1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t - 1}}
\epsilon_{t - 1}$$.
Considering the definition of $$q(\mathbf{x}_t | \mathbf{x}_{t - 1})$$, we have

$$
\begin{aligned}
\mathbf{x}_t & = \sqrt{1 - \beta_{t}} \mathbf{x}_{t - 1} + \sqrt{\beta_{t}} \epsilon_{t} \\
             & = \sqrt{1 - \beta_{t}} (
                                        \sqrt{\bar{\alpha}_{t - 1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t - 1}} \epsilon_{t - 1}
                                      ) + \sqrt{\beta_{t}} \epsilon_{t} \\
             & = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \beta_{t}} \sqrt{1 - \bar{\alpha}_{t - 1}} \epsilon_{t - 1} + \sqrt{\beta_{t}} \epsilon_t
\end{aligned}
$$

The right two items are the sum of two independent Gaussian variables. The mean
is 0, and the variance is 

$$
\alpha_t (1 - \bar{\alpha}_{t - 1}) + \beta_{t} = 1 - \bar{\alpha}_t
$$

which concludes the results.

## Backward pass
The backward is modeled as a Markov chain. $\mathbf{x}_{T}$ follows the standard Gaussian distribution.

$$
p(\mathbf{x}_{t-1}|\mathbf{x}_{t})\sim\mathcal{N}(x_{t-1};\mathbf{\mu}_{\theta}(\mathbf{x}_t,t),\mathbf{\Sigma}_{\theta}(\mathbf{x}_t,t))
$$

The parameter $\theta$ denotes all the learnable parameters to predict the mean
and the variance. Given a fixed $\theta$, we can find a marginal probability
distribution for $$\mathbf{x}_{0}$$,
starting from $$\mathbf{x}_{T}$$. 


## Training objective

The goal is to
find $$\theta$$ to maximize $$p(\mathbf{x}_{0})$$ when $$\mathbf{x}_0$$ follows
$$q(\mathbf{x}_0)$$.

Before moving forward, let's first derive the evidence lower bound.
If $f()$ is a convex function, we have $E[f(x)] \ge fE[x]$ based on
Jensen's inequality. Then,

$$
\log p(x) = \log \int p(x, z) dz = \log \int p(x, z) \frac{q(z)}{q(z)} dz = \log E_{z\sim q} \frac{p(x, z)}{q(z)}
\ge E_{z\sim q}\log \frac{p(x, z)}{q(z)}
$$

where $$q()$$ can be any probability function. The right side is called the
evidence lower bound as the left side is the probability of the observation,
which is called the evidence.

As $$q()$$ can be any function, we can change it to $$q(\mathbf{z}|\mathbf{x})$$. Also, let $$\mathbf{z}$$
be $$\mathbf{x}_{1:T}$$. Then, we have

$$
\begin{aligned}
L = 
-E[\log p(\mathbf{x}_0)] \le -E [\log \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})}] 
& = - E
\log
\frac{
    p(\mathbf{x}_T) \prod_{t=1}^{T} p(\mathbf{x}_{t - 1}|\mathbf{x}_{t})
}{
    \prod_{t = 1}^{T} q(x_t|x_{t - 1})
} \\
& =
E\left[
-\log p(\mathbf{x}_{T}) - \sum_{t = 1}^{T} \log \frac{
    p(\mathbf{x}_{t - 1}|\mathbf{x}_{t})
}{
    q(\mathbf{x}_t|\mathbf{x}_{t - 1})
}
\right]
\end{aligned}
$$

This corresponds to Eqn (1) of Reference [2] or Eqn (3) of Reference [1].

Considering $$q(\mathbf{x}_t)$$ is a Markov process, we have

$$
q(\mathbf{x}_t | \mathbf{x}_{t - 1}) = q(\mathbf{x}_t | \mathbf{x}_{t - 1}, \mathbf{x}_{0}) = 
\frac{
q(\mathbf{x}_t, \mathbf{x}_{t - 1} | \mathbf{x}_{0})
}{
q(\mathbf{x}_{t - 1} | \mathbf{x}_0)
} = 
\frac{
q(\mathbf{x}_{t} | \mathbf{x}_{0}) q(\mathbf{x}_{t - 1} | \mathbf{x}_{t}, \mathbf{x}_0)
}{
q(\mathbf{x}_{t - 1} | \mathbf{x}_0)
}, t > 1
$$

Thus,

$$
\begin{aligned}
L & = E\left[
- \log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{
p(\mathbf{x}_{t - 1} | \mathbf{x}_t)
}{
q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_0)
} \frac{
q(\mathbf{x}_t | \mathbf{x}_0)
}{
q(\mathbf{x}_{t - 1} | \mathbf{x}_0)
}
- \log \frac{
p(\mathbf{x}_0 | \mathbf{x}_1)
}{
q(\mathbf{x}_1 | \mathbf{x}_0)
}
\right] \\
& = E\left[
- \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_{T} | \mathbf{x}_0)}
- \sum_{t > 1} \log \frac{
p(\mathbf{x}_{t - 1} | \mathbf{x}_t)
}{
q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_0)
} 
- \log p(\mathbf{x}_0 | \mathbf{x}_1)
\right] \\
& = E\left[
\underbrace{KL(q(\mathbf{x}_T | \mathbf{x}_0) \| p(\mathbf{x}_T))}_{L_{T}} + 
\sum_{t > 1} \underbrace{KL(q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_0) \| p(\mathbf{x}_{t - 1} | \mathbf{x}_t))}_{L_{t - 1}}
- \underbrace{\log p(\mathbf{x}_0 | \mathbf{x}_1)}_{L_{0}}
\right]
\end{aligned}
$$

This corresponds to Eqn (5) of Reference [1].
The loss in $L_{T}$ contains no learnable parameters, recalling that all
learnable parameters are $\theta$.
To calculate $L_{t - 1}$, let's first
calculate $q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_0)$.

$$
q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_0) = \frac
{
    q(\mathbf{x}_{t - 1}, \mathbf{x}_t | \mathbf{x}_0)
}
{
    q(\mathbf{x}_t | \mathbf{x}_0)
}
= \frac
{
    q(\mathbf{x}_{t - 1} | \mathbf{x}_0) q(\mathbf{x}_t | \mathbf{x}_{t - 1}, \mathbf{x}_0)
}
{
    q(\mathbf{x}_t | \mathbf{x}_0)
}
= \frac
{
    q(\mathbf{x}_{t - 1} | \mathbf{x}_0) q(\mathbf{x}_t | \mathbf{x}_{t - 1})
}
{
    q(\mathbf{x}_t | \mathbf{x}_0)
}
$$

The last equation is from the Markov property, i.e. 
$$q(\mathbf{x}_t | \mathbf{x}_{t - 1}, \mathbf{x}_0) = q(\mathbf{x}_t | \mathbf{x}_{t - 1})$$.
The denorminator contains no $\mathbf{x}_{t - 1}$, and the numerator has 2 Gaussian distribution.
Thus, the
result must also follow the Gaussian distribution.
We only need to calculate the exponential part (excluding $$-1/2$$). That is,

$$
\begin{aligned}
& (\mathbf{x}_{t - 1} - \sqrt{\bar{\alpha}_{t - 1}}\mathbf{x}_0)^{'} (1 - \bar{\alpha}_{t - 1})^{-1}
(\mathbf{x}_{t - 1} - \sqrt{\bar{\alpha}_{t - 1}}\mathbf{x}_0) + 
(\mathbf{x}_{t} - \sqrt{\alpha_{t}}\mathbf{x}_{t - 1})^{'} \beta_{t}^{-1} (\mathbf{x}_{t} - \sqrt{\alpha_{t}}\mathbf{x}_{t - 1})
\\
= &
(
    (1 - \bar{\alpha}_{t - 1})^{-1} + \alpha_t / \beta_{t}
)   \|\mathbf{x}_{t - 1}\|^2 
- 2
(
 \sqrt{\bar{\alpha}_{t - 1}}\mathbf{x}_0^{'} / (1 - \bar{\alpha}_{t - 1}) + \sqrt{\alpha_t} / \beta_{t} \mathbf{x}_{t}^{'}
) \mathbf{x}_{t - 1} + C
\end{aligned}
$$

Thus, we can conclude the mean and the convariance are

$$
\begin{aligned}
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) 
& = \frac
{
    \sqrt{\bar{\alpha}_{t - 1}}\mathbf{x}_0^{'} / (1 - \bar{\alpha}_{t - 1}) + \sqrt{\alpha_t} / \beta_{t} \mathbf{x}_{t}
}
{
    (1 - \bar{\alpha}_{t - 1})^{-1} + \alpha_t / \beta_{t}
}
= \frac
{
    \beta_t \sqrt{\bar{\alpha}_{t - 1}} \mathbf{x}_0 + \sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t - 1}) \mathbf{x}_t
}
{
    1 - \bar{\alpha}_{t}
}
\\
\tilde{\beta}_{t} & = ((1 - \bar{\alpha}_{t - 1})^{-1} + \alpha_t / \beta_t)^{-1}
= \frac{
    \beta_{t} (1 - \bar{\alpha}_{t - 1})
}{
    1 - \bar{\alpha}_t
}
\end{aligned}
$$

This corresponds to Eqn (6) (7) of Reference [1].

Now let's get back to the loss of $L_{t - 1}$, which is a KL loss between two
Gaussian distribution.
The first distribution is $$r(\mathbf{x}) = q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_0)$$,
which is constant (does not depend on the learnable parameters).
The second distribution is $$
s_\theta(\mathbf{x}) = p(\mathbf{x}_{t - 1} | \mathbf{x}_t) = \mathcal{N}(
\mathbf{x}_{t - 1}; \mu_{\theta}(\mathbf{x}_t, t), \Sigma_{\theta}(\mathbf{x}_t, t)
)
$$.
We assume $$\Sigma_{\theta}(\mathbf{x}_t, t) = \sigma_{t}^{2}\mathbf{I}$$ following Reference [1].
The parameter $$\sigma_t$$ is also fixed, not learnable.

$$
\begin{aligned}
L_{t - 1} 
& = KL(r(\mathbf{x}) \| s_\theta(\mathbf{x})) \\
& = \int r(\mathbf{x}) \log
    \frac{r(\mathbf{x})}{s_\theta(\mathbf{x})} d \mathbf{x} 
\\
& = -\int r(\mathbf{x}) \log s_\theta(\mathbf{x}) d \mathbf{x} + C_1
\\
& = -\int r(\mathbf{x}) \log \exp\left(
- \frac{1}{2\sigma_{t}^{2}}\|\mathbf{x} - \mu_{\theta}(\mathbf{x}_t, t)\|^2
\right) d \mathbf{x} + C_2
\\
& = -\int r(\mathbf{x}) \log \exp\left(
- \frac{1}{2\sigma_{t}^{2}}\| \mathbf{x} - \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) +
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) - \mu_{\theta}(\mathbf{x}_t, t)\|^2
\right) d \mathbf{x} + C_2
\\
& = -\int r(\mathbf{x}) \left(
- \frac{1}{2\sigma_{t}^{2}} \| \mathbf{x} - \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) +
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) - \mu_{\theta}(\mathbf{x}_t, t)\|^2
\right) d \mathbf{x} + C_2
\\
& = -\int r(\mathbf{x}) \left(
- \frac{1}{2\sigma_{t}^{2}} \|
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) - \mu_{\theta}(\mathbf{x}_t, t)\|^2
\right) d \mathbf{x} + C_3
\\
& = 
\frac{1}{2\sigma_{t}^{2}} \|
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) - \mu_{\theta}(\mathbf{x}_t, t)\|^2
+ C_3
\end{aligned}
$$

This corresponds to Eqn (8) of Reference [1].
With $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 -
\bar{\alpha}_t} \epsilon
$$, ($$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$), we have

$$
\begin{aligned}
    \mathbf{x}_0                                &= \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\epsilon) \\
    \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)   &= 
\frac
{ \beta_t \sqrt{\bar{\alpha}_{t - 1}} }
{ 1 - \bar{\alpha}_{t} }
\mathbf{x}_0 +  \frac
{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t - 1})}
{1 - \bar{\alpha}_{t}} \mathbf{x}_t \\
& = 
\frac
{ \beta_t \sqrt{\bar{\alpha}_{t - 1}} }
{ 1 - \bar{\alpha}_{t} }\frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\epsilon) + 
\frac
{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t - 1})}
{1 - \bar{\alpha}_{t}} \mathbf{x}_t  \\
& = \frac{1}{\sqrt{\alpha_t}}\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha_t}} \epsilon \\
& = \frac{1}{\sqrt{\alpha_t}} (
\mathbf{x}_t - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha}_t}} \epsilon
)
\end{aligned}
$$

Although we need to learn $$\mu_\theta(\mathbf{x}_t, t)$$, we can
reparameterize it with a learnable $$\epsilon_\theta(\mathbf{x}_t, t)$$ as
follows:

$$
\mu_{\theta}(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} (
\mathbf{x}_t - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t)
)
$$

This is Eqn (11) of Reference [1]. Then, we have the loss $$L_{t - 1}$$ as
follows (the constant value is removed)

$$
\begin{aligned}
L_{t - 1} & = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)} \|\epsilon - \epsilon_{\theta}(\mathbf{x}_t, t)\|^2 \\
\mathbf{x}_t & = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
\end{aligned}
$$


The $t$ is larger than 1, which means it is not applicable for the case of $t = 1$. The loss $L_{0}$ is specifically for the case of $t = 1$, which is not in a similar form 
of $L_{t - 1}$. However, in the paper of Reference [1], the author suggests to
simplify the training objectives by the following

$$
\begin{aligned}
L_\text{simple} & = E\|\epsilon - \epsilon_{\theta}(\mathbf{x}_t, t)\|^2 \\
\mathbf{x}_t & = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
\end{aligned}
$$

In the expectation of $$L_\text{simple}$$, the timestep $t$ can start from 1,
rather than from 2. The coefficient is also removed, which means a weighted sum
of the original $L_{t - 1}$.


Thus, in summary, during training, we 1) randomly sample a timestep $$t$$, 2)
sample a noise $$\epsilon$$, 3) calculate $\mathbf{x}_t$, 4) predict the noise
based on our network $$\epsilon_{\theta}(\mathbf{x}_t, t)$$, 5) calculate the
difference as the loss function.

During inference, we can simply get $\mu_{\theta}$ based on $\epsilon_{\theta}$
to predict $\mathbf{x}_{t - 1}$ from $\mathbf{x}_t$, starting from $$\mathbf{x}_{T}$$.

## References  

[1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.

[2] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. and Ganguli, S., 2015, June. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning (pp. 2256-2265). PMLR.


