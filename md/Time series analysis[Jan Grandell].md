#TSA by JG

[TOC]

## Intro

### Stationarity

### Trends & Seasonal Components

#### No Seasonal Components
#### Trend and Seasonality

## 

### Autocovariance

#### Strict stationarity

#### spectral density

### Time Series Model TSM

## 

### Estimation

#### Estimation of $\mu$

#### Estimation of $\gamma(\cdot)$ and $\rho(\cdot)$

### Prediction

#### inference

#### Prediction of Random Variables PRV

## 

### Further Prediction

#### Further PRV

#### Predition for stationary time series PSTS

## 

### wold decomposition

### Partial Correlation

#### PA Partial Autocorrelation

### ARMA Processes

#### ACVF and how to cal

#### Prediction of ARMA

## 

### Spectral Analysis

#### Spectral Distribution

#### Spectral Representation of a time series

### Predition in the frequency domain PiFD 

#### interpolation and detection

### The Itˆo intergral

## 

### Estimation of the spectral density

#### periodogram

#### Smoothing the Periodogram SP

### Linear Filters

#### ARMA Processes

## 

### Estimation for ARMA models

#### Yule-Walker estimation

#### Burg's algorithm

#### innovation algorithm

#### hannan-Rissanen algorithm

#### Maximum Likelihood and Lead Square Estimation

#### Order Selection

##  

### unit roots

### Multivariate Time Series MTS

## 

### Financial Time Series

## 

### Kalman Filtering

#### State-Space Representation

#### Prediction of Multivariate random variables

## Appendix

### A.1 Stochastic Processes

**Definition (Stochastic Process)** A stochastic process is a *family of random variables* $\{ X_t \, , \, t \in T\}$ *defined on a probability space* $(\Omega,\mathcal{F},P)$ as follows:

* $\Omega$ is a set
* $\mathcal{F}$ is a $\sigma$-field i.e.
    * (a) $\emptyset \,\in \, \mathcal{F}$
    * (b) $A_i \,\in\, \mathcal{F} \quad \forall\, i \,\in\, I$ then  $\bigcup_{i} A_i \,\in\, \mathcal{F}$
    * (c) $A \,\in\, \mathcal{F}$ then $A^c \,\in\, \mathcal{F}$
* $P$ is a function $\mathcal{F} \rightarrow [0,1]$ satisfying:
    * $P(\Omega) = 1$
    * $A_i \,\in\, \mathcal{F} \quad \forall\, i \,\in\, I$ and $A_i \cap A_j = \emptyset \quad \forall\, i \,\in\, I$ then $P(\bigcup_{i}A_i) = \sum_{i} P(A_i)$

**proposition** $P(A) + P(A^c) = 1$

There are definitions on Cholton's Style:

($X$ is *random variable*)  iff  (($X$ is a function $\Omega \to \mathbb{R}$) and ($\{\omega \,\in\, \Omega : X(\omega) \leq x\} \,\in\, \mathcal{F}$ for all $x \,\in\, \mathbb{R}$))

($\{X_t \,,\, t \in T\}$ is *stochastic process*)  iff  ($X_t$ is *random variable* for all $t \,\in\, T$)

and T is called *index* or *parameter set*

($\{X_t \,,\, t \in T\}$ is a *time series*)  iff  ($T \subset \mathbb{Z}$)

**Definition (sample-path)** The functions $\{ X_t(\omega) \,,\, \omega \,\in\, \Omega\}$ on T are called realizations or *sample-path* of the process $\{X_t \,,\, t \,\in\, T\}$

$F_{X}, x \to P(X \leq x)$ is called the distribution function of a random variable X
$F_{(\cdot)} : (\Omega \to \mathbb{R}) \to (\mathbb{R} \to [0,1])$ is called the distribution

The case in higher demension is similar.

$X = (X_1, \,\cdots\,, X_n)^\top$ is a $n$-dim *random variable*, $X_i$ is a *random variable* for $1 \,\leq\, i \,\leq\, n$

**Definition (The distribution of a stochastic process)** let
    $\mathcal{T} := \{t \in T^n \,:\, t_i<t_j\}$
The (finitie-dimensional) distribution function are the family $\{F_t(\cdot)\,,\, t \,\in\, \mathcal{T}\}$, $F_t(x) = P(X_{t_1} \leq x_1, \cdots , X_{t_n} \leq x_n) \quad t \,\in\, T^n \,,\, x \in \mathbb{R}$
The distribution of $\{X_t \,,\, t \in T\}$ is the family $\{F_t(\cdot)\,,\, t \,\in\, \mathcal{T}\}$

obviously $\mathcal{T}$ is a simplex
In a way, we can say:"$t \in T$ is a *series of time*". So the $F_t$ is a distribution of a *series of time* of random variable i.e. the distribution of *stochastic process*. In conviencity, $F_t \sim \mathcal{T} \sim X_t \sim n$, where the symbol $\sim$ readed as 'related to'.

**Theorem (Kolmogorov's existence theorem)** The family $F_t(\cdot)\,,\, t \,\in\, \mathcal{T}$ are the distribution functions of some stochastic process 
### A.2 Hilbert Spaces