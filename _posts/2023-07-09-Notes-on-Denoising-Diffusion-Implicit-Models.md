---
layout: post
comments: false
title: Notes on Denoising Diffusion Implicit Models
author: Jianfeng Wang
---


## Forward pass
In DDPM, the forward pass is a Markov chain, but in the model of DDIM, it is
not.

Given the observation $$\mathbf{x}_0$$ following an unknown distribution
$$q(\mathbf{x}_0)$$ and a timestep $$T$$ for a Gaussian distribution 

$$
q(\mathbf{x}_{T} | \mathbf{x}_0) =\mathcal{N}(\sqrt{\bar{\alpha}_T}\mathbf{x}_0; (1 - \bar{\alpha}_T) \mathbf{I}),
$$

we define the following

$$
q(\mathbf{x}_{1:T} | \mathbf{x}_0) = q(\mathbf{x}_T | \mathbf{x}_0) \prod_{t =
2}^{T} q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_0).
$$

Note that, the paper of DDIM uses $$\alpha_{t}$$, which is actually closely related to $$\bar{\alpha}_{t}$$ in the DDPM paper.
Thus, we use $$\bar{\alpha}_t$$ here.
The process says that 1) we have $$\mathbf{x}_0$$,
2) draw $$\mathbf{x}_T$$ from $$q(\mathbf{x}_T | \mathbf{x}_0)$$,
3) draw  $$\mathbf{x}_{T - 1}$$ based on $$q(\mathbf{x}_{T - 1} | \mathbf{x}_{T}, \mathbf{x}_0)$$,
4) draw  $$\mathbf{x}_{T - 2}$$ based on $$q(\mathbf{x}_{T - 2} | \mathbf{x}_{T - 1}, \mathbf{x}_0)$$,
until we draw $$\mathbf{x}_1$$ based on $$q(\mathbf{x}_1 | \mathbf{x}_2,
\mathbf{x}_{0})$$. The probability of $$q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_0)$$ is defined as

$$
q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}
(
    \sqrt{\bar{\alpha}_{t - 1}} \mathbf{x}_0 
    + 
    \sqrt{1 - \bar{\alpha}_{t - 1} - \sigma_t^2}
    \frac
    {
        \mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0
    }
    {
        \sqrt{1 - \bar{\alpha}_{t}}
    },
    \sigma_{t}^{2} \mathbf{I}
).
$$

This definition leads to the following

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}
(
    \sqrt{\bar{\alpha}_t} \mathbf{x}_0;
    (1 - \bar{\alpha}_t) \mathbf{I}
).
$$

Let's verify it through induction.
The equation holds for $$t = T$$ because of the definition. Let's say it also holds
for some $$t$$ starting from $$T$$.
Then,
we only need to calculate $$q(\mathbf{x}_{t - 1} | \mathbf{x}_0)$$.
Considering the equation holds for $$q(\mathbf{x}_t | \mathbf{x}_0)$$, we
have

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t
$$

where $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$. Given $$\mathbf{x}_0$$ and
$$\mathbf{x}_t$$, we can draw $$\mathbf{x}_{t - 1}$$ through $$q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \mathbf{x}_)$$:

$$
\mathbf{x}_{t - 1} = \sqrt{\bar{\alpha}_{t - 1}} \mathbf{x}_0 + 
    \sqrt{1 - \bar{\alpha}_{t - 1} - \sigma_t^2}
    \frac
    {
        \mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0
    }
    {
        \sqrt{1 - \bar{\alpha}_{t}}
    } + \sigma_{t}^2 \epsilon_{t - 1}.
$$

By substituting $$\mathbf{x}_t$$, we have

$$
\mathbf{x}_{t - 1} = \sqrt{\bar{\alpha}_{t - 1}} \mathbf{x}_0 +
    \sqrt{1 - \bar{\alpha}_{t - 1} - \sigma_t^2} \epsilon_{t} + \sigma_{t}^{2} \epsilon_{t - 1}.
$$

Thus, we can easily conclude $$q(\mathbf{x}_{t - 1} | \mathbf{x}_{0})$$ also
follows the Gaussian distribution. The mean is $$\sqrt{\bar{\alpha}_{t - 1}}$$,
and the variance is $$1 - \bar{\alpha}_{t - 1} - \sigma_{t}^2 + \sigma_{t}^{2} = 1 - \bar{\alpha}_{t - 1}$$.

An interesting property is that the probability $$q(\mathbf{x}_{t}|\mathbf{x}_{0})$$ is 
un-related with any $$\sigma$$.


## Backward pass for a generative process
Recall the form of $$q(\mathbf{x}_T | \mathbf{x}_t)$$, if we design $$\bar{\alpha}_{T}$$ close to 0, it 
will be close to a normal Gaussian distribution. Thus, we can draw a
sample from normal distribution as an approximation of $$\mathbf{x}_T$$.
For any $$t$$, we can draw $$\mathbf{x}_t$$ based on a random sample
$$\epsilon_{t}$$.
A generative network model is designed to learn such noise by $$\epsilon{x}_{\theta}(\mathbf{x}_t)$$. Note that,
the network should also depend on the timestep $$t$$, and we ignore the
notation for simplicity.

With the predicted noise, we can estimate $$\mathbf{x}_0$$ as

$$
\hat{\mathbf{x}}_0(\mathbf{x}_t) = \frac
{
    \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_{\theta}(\mathbf{x}_t)
}
{
    \sqrt{\alpha_{t}}
}.
$$

Then, we can draw another sample of $$\mathbf{x}_{t - 1}$$ by the probability
of $$q(\mathbf{x}_{t - 1} | \mathbf{x}_t, \hat{\mathbf{x}}_0)$$.
This process repeats until we reach $\mathbf{x}_{1}$. Then, the original
$$\mathbf{x}_0$$ is estimated by $$\hat{\mathbf{x}}_0(\mathbf{x}_1)$$.

The key is to train $$\epsilon_{\theta}(\mathbf{x}_t)$$, which can follow
exactly the same process of DDPM. Thus, the training process between DDIM and
DDPM is the same, and the difference is only in inference.

## $$\sigma_{t} = 0$$ for DDIM
When $$\sigma_{t} = 0$$, the process is called DDIM. That is, in the forward
pass, we draw $$\mathbf{x}_T$$ based on $$\mathbf{x}_0$$, with some random
noise $$\epsilon_{T}$$. Then, we will draw all other $$\mathbf{x}_t$$ in a
deterministic way, rather than in a random way, as $$\sigma_t$$ is always 0.

In the backward generative process, we first draw a random sample from normal
distribution as an approximation of $$\mathbf{x}_{T}$$. Then, all other
samples will be drawn also in the deterministic way.

## Appropriate $$\sigma_{t}$$ for DDPM
When $$\sigma_{t}$$ is larger than 0, we will instroduce some noise in both the
forward pass and the backward pass. One special case is when 

$$
\sigma_{t} = \sqrt{
\frac
{
    1 - \bar{\alpha}_{t - 1}
}
{
    1 - \bar{\alpha}_{t}
}
(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t - 1}})
}.
$$

Recall DDPM's inference process, we know both DDPM and DDIM drive $$\mathbf{x}_{t - 1}$$ based on $$\hat{\mathbf{x}}_0$$ and $$\mathbf{x}_t$$.
In DDIM, the variance is 

$$
\sigma_t^2 = 
\frac
{
    1 - \bar{\alpha}_{t - 1}
}
{
    1 - \bar{\alpha}_{t}
}
(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t - 1}})=
\frac
{
    1 - \bar{\alpha}_{t - 1}
}
{
    1 - \bar{\alpha}_{t}
}\beta_{t}
$$

Recall that in DDPM, $$\bar{\alpha}_{t} = \prod_{i = 1}^{t} \alpha_{i} = \prod_{i = 1}^{t} (1 - \beta_{i})$$. This variance is exactly the same 
as the Eqn (7) of the DDPM paper (reference [1]).

To derive the mean, let's first calculate 

$$
\begin{aligned}
1 - \bar{\alpha}_{t - 1} - \sigma_{t}^2 & = 
1 - \bar{\alpha}_{t - 1} - 
\frac
{
    1 - \bar{\alpha}_{t - 1}
}
{
    1 - \bar{\alpha}_{t}
}
\frac
{
    \bar{\alpha}_{t - 1} - \bar{\alpha}_{t}
}
{
    \bar{\alpha}_{t - 1}
} \\
& = 
\frac
{
    (1 - \bar{\alpha}_t - \bar{\alpha}_{t - 1} + \bar{\alpha}_t\bar{\alpha}_{t - 1}) \bar{\alpha}_{t - 1} - (1 - \bar{\alpha}_{t - 1}) (\bar{\alpha}_{t - 1} - \bar{\alpha}_{t})
}
{
    (1 - \bar{\alpha}_t) \bar{\alpha}_{t - 1}
} \\
& = 
\frac
{
    - 2 \bar{\alpha}_{t} \bar{\alpha}_{t - 1} + \bar{\alpha}_{t} \bar{\alpha}_{t - 1}^{2} + \bar{\alpha}_t
}
{
    (1 - \bar{\alpha}_t) \bar{\alpha}_{t - 1}
}
\\
& = 
\frac
{
    \bar{\alpha}_t (1 - \bar{\alpha}_{t - 1})^2
}
{
    (1 - \bar{\alpha}_t) \bar{\alpha}_{t - 1}
}
\end{aligned}
$$


Then, the mean is

$$
\begin{aligned}
&
\sqrt{\bar{\alpha}_{t - 1}} \mathbf{x}_0 
+ 
\sqrt{1 - \bar{\alpha}_{t - 1} - \sigma_t^2}
\frac
{
    \mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0
}
{
    \sqrt{1 - \bar{\alpha}_{t}}
}
\\
= & 
(
\sqrt{\bar{\alpha}_{t - 1}} - 
\sqrt{1 - \bar{\alpha}_{t - 1} - \sigma_t^2}
\frac
{
    \sqrt{\bar{\alpha}_t}
}
{
    \sqrt{1 - \bar{\alpha}_t}
}
)
\mathbf{x}_0
+
\frac
{
    \sqrt{1 - \bar{\alpha}_{t - 1} - \sigma_t^2}
}
{
    \sqrt{1 - \bar{\alpha}_{t}}
}\mathbf{x}_t
\\
= & 
(
\sqrt{\bar{\alpha}_{t - 1}} - 
\frac
{
    \sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t - 1})
}
{
    \sqrt{1 - \bar{\alpha}_t} \sqrt{\bar{\alpha}_{t - 1}}
}
\frac
{
    \sqrt{\bar{\alpha}_t}
}
{
    \sqrt{1 - \bar{\alpha}_t}
}
)
\mathbf{x}_0
+
\frac
{
    \sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t - 1})
}
{
    \sqrt{1 - \bar{\alpha}_{t}}
    \sqrt{1 - \bar{\alpha}_t} \sqrt{\bar{\alpha}_{t - 1}}
}\mathbf{x}_t
\\
= & 
(
\sqrt{\bar{\alpha}_{t - 1}} - 
\frac
{
    \bar{\alpha}_t (1 - \bar{\alpha}_{t - 1})
}
{
    (1 - \bar{\alpha}_t) \sqrt{\bar{\alpha}_{t - 1}}
}
)
\mathbf{x}_0
+
\frac
{
    \sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t - 1})
}
{
    (1 - \bar{\alpha}_{t}) \sqrt{\bar{\alpha}_{t - 1}}
}\mathbf{x}_t
\\
= & 
(
\frac
{
    \bar{\alpha}_{t - 1} - \bar{\alpha}_{t}
}
{
    (1 - \bar{\alpha}_t) \sqrt{\bar{\alpha}_{t - 1}}
}
)
\mathbf{x}_0
+
\frac
{
    \sqrt{\alpha_t} (1 - \bar{\alpha}_{t - 1})
}
{
    1 - \bar{\alpha}_{t}
}\mathbf{x}_t
\\
= & 
(
\frac
{
    \sqrt{\bar{\alpha}_{t - 1}} \beta_{t}
}
{
    1 - \bar{\alpha}_t
}
)
\mathbf{x}_0
+
\frac
{
    \sqrt{\alpha_t} (1 - \bar{\alpha}_{t - 1})
}
{
    1 - \bar{\alpha}_{t}
}\mathbf{x}_t
\end{aligned}
$$

This is exactly the same as the mean of Eqn (7) of DDPM paper. Thus, the
inference process is exactly the same as DDPM.

## References

[1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.

[2] Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).
