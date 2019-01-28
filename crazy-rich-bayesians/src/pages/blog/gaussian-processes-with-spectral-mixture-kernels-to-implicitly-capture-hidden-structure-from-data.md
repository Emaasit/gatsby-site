---
templateKey: article-page
title: Gaussian Processes with Spectral Mixture Kernels to Implicitly Capture Hidden Structure from Data
date: 2018-03-19T19:29:55.624Z
cover: /img/gatsby-image-workarounds.jpeg
tags:
  - GaussianProcess
  - SpectralMixtureKernels
meta_title: Gaussian Processes with Spectral Mixture Kernels to Implicitly Capture Hidden Structure from Data
meta_description: >-
  The scientific field of insider-threat detection often lacks sufficient amounts of time-series training data for the purpose of scientific discovery. Moreover, the available limited data are quite noisy. For instance Greitzer and Ferryman (2013) state that ”ground truth” data on actual insider behavior is typically either not available or is limited.
---
**(Note: Cross-posted with the [Haystax Technology Blog](https://haystax.com/blog/).)**

The scientific field of insider-threat detection often lacks sufficient amounts of time-series training data for the purpose of scientific discovery. Moreover, the available limited data are quite noisy. For instance Greitzer and Ferryman (2013) state that ”ground truth” data on actual insider behavior is typically either not available or is limited. In some cases, one might acquire real data, but for privacy reasons, there is no attribution of any individuals relating to abuses or offenses i.e., there is no ground truth. The data may contain insider threats, but these are not identified or knowable to the researcher (Greitzer and Ferryman, 2013; Gheyas and Abdallah, 2016).

## The problem
Having limited and quite noisy data for insider-threat detection presents a major challenge when estimating time-series models that are robust to overfitting and have well-calibrated uncertainty estimates. Most of the current literature in time-series modeling for insider-threat detection is associated with two major limitations.

First, the methods involve visualizing the time series for noticeable structure and patterns such as periodicity, smoothness, growing/decreasing trends and then hard-coding these patterns into the statistical models during formulation. This approach is suitable for large datasets where more data typically provides more information to learn expressive structure. Given limited amounts of data, such expressive structure may not be easily noticeable. For instance, the figure below shows monthly attachment size in emails (in Gigabytes) sent by an insider from their employee account to their home account. Trends such as periodicity, smoothness, growing/decreasing trends are not easily noticeable.

<img src="https://github.com/Emaasit/long-range-extrapolation/blob/dev/blog/data-emails.png?raw=true" width="600" height="200" />

Second, most of the current literature focuses on parametric models that impose strong restrictive assumptions by pre-specifying the functional form and number of parameters. Pre-specifying a functional form for a time-series model could lead to either overly complex model specifications or simplistic models. It is difficult to know *a priori* the most appropriate function to use for modeling sophisticated insider-threat behavior that involve complex hidden patterns and many other influencing factors.

### Source code
For the impatient reader, two options are provided below to access the source code used for empirical analyses:

1. Most of the code is not shown here to keep the post concise but the code, data and results can be found [here on GitHub](https://github.com/Emaasit/long-range-extrapolation).

2. [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Emaasit/long-range-extrapolation/master?urlpath=lab) Click this icon to open the notebooks in a web browser and explore the code and data without downloading the project and installing any software.


## Related Work and Limitations
This approach is associated with two limitations. First, given that such trends may not be noticeable in small data, it is difficult to explicitly incorporate expressive structure into the statistical models during formulation.  Second, it is difficult to know *a priori* the most appropriate functional form to use. 

## Data Science Questions
Given the above limitations in the current state-of-art, this study formulated the following three Data Science questions. Given limited and quite noisy time-series data for insider-threat detection, is it possible to perform:

1. pattern discovery without hard-coding trends into statistical models during formulation?

2. model estimation that precludes pre-specifying a functional form?

3. model estimation that is robust to overfitting and has well-calibrated uncertainty estimates? 

## Hypothesis
To answer these three Data Science questions and address the above-described limitations, this study formulated the following hypothesis:
<blockquote>This study hypothesizes that by leveraging current state-of-the-art innovations in Nonparametric Bayesian methods, such as Gaussian processes, it is possible to perform pattern discovery without prespecifying functional forms and hard-coding trends into statistical models.</blockquote>

## Methodology
To test the above hypothesis, a nonparametric Bayesian approach was proposed to implicitly capture hidden structure from time series having limited data. The proposed model, a Gaussian process with a spectral mixture kernel, precludes the need to pre-specify a functional form and hard code trends, is robust to overfitting and has well-calibrated uncertainty estimates.

Mathematical details of the proposed model formulation are described in a corresponding paper that can be found on arXiv through the link below:

* Emaasit, D. and Johnson, M. (2018). [Capturing Structure Implicitly from Noisy Time-Series having Limited Data](https://arxiv.org/abs/1803.05867). arXiv preprint arXiv:1803.05867.

A Brief description of the fundamental concepts of the proposed methodology are as follows. Consider for each data point, $latex i$, that $latex y_i$ represents the attachment size in emails sent by an insider to their home account and $latex x_i$ is a temporal covariate such as month. The task is to estimate a latent function $latex f$, which maps input data, $latex x_i$, to output data $latex y_i$ for $latex i$ = 1, 2, $latex \ldots{}$, $latex N$, where $latex N$ is the total number of data points. Each of the input data $latex x_i$ is of a single dimension $latex D = 1$, and $latex \textbf{X}$ is a $latex N$ x $latex D$ matrix with rows $latex x_i$.

<img class="size-medium wp-image-6429 aligncenter" src="http://haystax.com/wp-content/uploads/2018/03/gp-pgm-352x300.png" alt="" width="352" height="200" />

The observations are assumed to satisfy:
\begin{equation}\label{eqn:additivenoise}
y_i = f(x_i) + \varepsilon, \quad where \, \, \varepsilon \sim \mathcal{N}(0, \sigma_{\varepsilon}^2)
\end{equation}
The noise term, $latex \varepsilon$, is assumed to be normally distributed with a zero mean and variance, $latex \sigma_{\varepsilon}^2$. Latent function $latex f$ represents hidden underlying trends that produced the observed time-series data.

Given that it is difficult to know $latex \textit{a priori}$ the most appropriate functional form to use for $latex f$, a prior distribution, $latex p(\textbf{f})$, over an infinite number of possible functions of interest is formulated. A natural prior over an infinite space of functions is a Gaussian process prior (Williams and Rasmussen, 2006). A GP is fully parameterized by a mean function, $latex \textbf{m}$, and covariance function, $latex \textbf{K}_{N,N}$, denoted as:
\begin{equation}\label{eqn:gpsim}
\textbf{f} \sim \mathcal{GP}(\textbf{m}, \textbf{K}_{N,N}),
\end{equation}

The posterior distribution over the unknown function evaluations, $latex \textbf{f}$, at all data points, $latex x_i$, was estimated using Bayes theorem as follows:
\begin{equation}\label{eqn:bayesinfty}
\begin{aligned}
p(\textbf{f} \mid \textbf{y},\textbf{X}) &amp;= \frac{p(\textbf{y} \mid \textbf{f}, \textbf{X}) \, p(\textbf{f})}{p(\textbf{y} \mid \textbf{X})} = \frac{p(\textbf{y} \mid \textbf{f}, \textbf{X}) \, \mathcal{N}(\textbf{f} \mid \textbf{m}, \textbf{K}_{N,N})}{p(\textbf{y} \mid \textbf{X})},
\end{aligned}
\end{equation}
where:

$latex p(\textbf{f}\mid \textbf{y},\textbf{X})$ = the posterior distribution of functions that best explain the email-attachment size, given the covariates
$latex p(\textbf{y} \mid \textbf{f}, \textbf{X})$ = the likelihood of email-attachment size, given the functions and covariates
$latex p(\textbf{f})$ = the prior over all possible functions of email-attachment size
$latex p(\textbf{y} \mid \textbf{X})$ = the data (constant)

This posterior is a Gaussian process composed of a distribution of possible functions that best explain the time-series pattern.

## Experiments

### The setup
Let's first install some python packages that we shall use for our analysis. Also we shall set up our plotting requirements.


```python
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('notebook', font_scale = 1.1)
np.random.seed(12345)
rc = {'xtick.labelsize': 40, 'ytick.labelsize': 40, 'axes.labelsize': 40, 'font.size': 40, 'lines.linewidth': 4.0, 
      'lines.markersize': 40, 'font.family': "serif", 'font.serif': "cm", 'savefig.dpi': 200,
      'text.usetex': False, 'legend.fontsize': 40.0, 'axes.titlesize': 40, "figure.figsize": [24, 16]}
sns.set(rc = rc)
sns.set_style("darkgrid")
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import gpflow
from gpflowopt.domain import ContinuousParameter
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.acquisition import ExpectedImprovement
from gpflowopt.optim import StagedOptimizer, MCOptimizer, SciPyOptimizer  
from gpflowopt.design import LatinHyperCube
```

### Raw data and sample formation

The insider-threat data used for empirical analysis in this study was provided by the computer emergency response team (CERT) division of the software engineering institute (SEI) at Carnegie Mellon University. The particular insider threat focused on is the case where a known insider sent information as email attachments from their work email to their home email. The `pydata` software stack including packages such as `pandas`, `numpy`, `seaborn`, and others, was used for data manipulation and visualization

First, let's read in the data using `pandas`, view the first three records and the structure of the resulting `pandas` dataframe.


```python
email_filtered = pd.read_csv("../data/emails/email_filtered.csv", parse_dates=["date"])
email_filtered.head(n = 3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>user</th>
      <th>pc</th>
      <th>to</th>
      <th>cc</th>
      <th>bcc</th>
      <th>from</th>
      <th>activity</th>
      <th>size</th>
      <th>attachments</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{D0V4-N9KM15BF-0512LLVP}</td>
      <td>2010-01-04 07:36:48</td>
      <td>BTR2026</td>
      <td>PC-9562</td>
      <td>Thaddeus.Brett.Daniel@dtaa.com</td>
      <td>Zorita.Angela.Wilson@dtaa.com</td>
      <td>NaN</td>
      <td>Beau.Todd.Romero@dtaa.com</td>
      <td>Send</td>
      <td>23179</td>
      <td>NaN</td>
      <td>On November 25, general Savary was sent to the...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{L5E5-J1HB80OY-9539AOEC}</td>
      <td>2010-01-04 07:38:18</td>
      <td>BTR2026</td>
      <td>PC-9562</td>
      <td>Beau.Todd.Romero@dtaa.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Marsh_Travis@raytheon.com</td>
      <td>View</td>
      <td>17047</td>
      <td>NaN</td>
      <td>Early in the morning of May 27, a boat crossed...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{Q4V7-V6BR00TZ-5209UVDX}</td>
      <td>2010-01-04 07:53:35</td>
      <td>BTR2026</td>
      <td>PC-9562</td>
      <td>Bianca-Clark@optonline.net</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Beau_Romero@aol.com</td>
      <td>Send</td>
      <td>26507</td>
      <td>NaN</td>
      <td>The Americans never held up their side of the ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
email_filtered.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11920 entries, 0 to 11919
    Data columns (total 12 columns):
    id             11920 non-null object
    date           11920 non-null datetime64[ns]
    user           11920 non-null object
    pc             11920 non-null object
    to             11920 non-null object
    cc             6101 non-null object
    bcc            593 non-null object
    from           11920 non-null object
    activity       11920 non-null object
    size           11920 non-null int64
    attachments    3809 non-null object
    content        11920 non-null object
    dtypes: datetime64[ns](1), int64(1), object(10)
    memory usage: 1.1+ MB


Let's filter data for a particular known insider with user ID "CDE1846".


```python
df_insider = email_filtered[email_filtered["user"] == "CDE1846"]
df_insider.shape
```




    (3165, 12)




```python
emails_per_month = df_insider.resample(rule = "1M", on = "date").sum().reset_index()
emails_per_month["date"] = pd.to_datetime(emails_per_month["date"], format = "%Y-%m-%d")
emails_per_month.columns = ["ds", "y"]
emails_per_month
```




```python
fig, ax = plt.subplots()
sns.barplot(data = emails_per_month, x = "ds", y = "y", ax = ax)
ax.set_xticklabels(labels = emails_per_month["ds"], rotation = 45)
ax.set_xlabel("Months of the Year")
ax.set_ylabel("Number of Emails")
ax.set_title("Number of Emails sent Monthly");
```


![png](output_13_0.png)


Here, we look at the case where the insider email IP to their home account. The data is resampled per month and the anomalous behavior is clearly visible


```python
df_insider_non_org = df_insider[~df_insider['to'].str.contains('dtaa.com')]
df_insider_ewing = df_insider_non_org[df_insider_non_org['to'] == 'Ewing_Carlos@comcast.net']
df = df_insider_ewing.resample('1M', on='date').sum().reset_index()
df.columns = ["ds", "y"]
(df.y/1e6).describe()
df.y = df.y/1e6
```




    count     17.000000
    mean      13.042668
    std       27.010948
    min        0.181588
    25%        2.258068
    50%        4.749784
    75%        6.430011
    max      108.623858
    Name: y, dtype: float64




```python
from datetime import datetime
df["ds"] = df.apply(lambda x: datetime.date(x["ds"]), axis = 1)
```


```python
fig, ax = plt.subplots()
sns.barplot(data = df, x = "ds", y = "y")
ax.set_xticklabels(labels = df.ds, rotation = 45)
ax.set_xlabel("Time")
ax.set_ylabel("Number of Emails ($10^6$)");
# ax.set_title("Number of Emails sent Monthly");
```


![png](output_17_0.png)



```python
df = df.drop([14, 15, 16])
```


```python
test_size = 11
X_complete = np.array([df.index]).reshape((df.shape[0], 1)).astype('float64')
X_train = X_complete[0:test_size, ]
X_test = X_complete[test_size:df.shape[0], ]
Y_complete = np.array([df.y]).reshape((df.shape[0], 1)).astype('float64')
Y_train = Y_complete[0:test_size, ]
Y_test = Y_complete[test_size:df.shape[0], ]
D = Y_train.shape[1];
```


```python
fig, ax = plt.subplots()
ax.plot(X_train.flatten(),Y_train.flatten(), c ='b', marker = "o", label = "Training data")
ax.plot(X_test.flatten(),Y_test.flatten(), c='r', marker = "o", label = 'Test data')
ax.set_xticklabels(labels = df.ds, rotation = 45)
ax.set_xlabel('Time')
ax.set_ylabel('Total size of emails in GB')
plt.legend(loc = "best");
```


![png](output_20_0.png)


### Empirical analysis

This study used a Gaussian Process model with a Spectral Mixture (SM) kernel proposed by Wilson (2014). This is because the SM kernel is capable of capturing hidden structure with data without hard cording features in a kernel. Moreover, the SM kernel is capable of performing long-range extrapolation beyond available data.



```python
# Trains a model with a spectral mixture kernel, given an ndarray of 
# 2Q frequencies and lengthscales

Q = 10 # nr of terms in the sum
max_iters = 1000

def create_model(hypers):
    f = np.clip(hypers[:Q], 0, 5)
    weights = np.ones(Q) / Q
    lengths = hypers[Q:]

    kterms = []
    for i in range(Q):
        rbf = gpflow.kernels.RBF(D, lengthscales=lengths[i], variance=1./Q)
        rbf.lengthscales.transform = gpflow.transforms.Exp()
        cos = gpflow.kernels.Cosine(D, lengthscales=f[i])
        kterms.append(rbf * cos)

    k = np.sum(kterms) + gpflow.kernels.Linear(D) + gpflow.kernels.Bias(D)
    m = gpflow.gpr.GPR(X_train, Y_train, kern=k)
    return m

m = create_model(np.ones((2*Q,)))
```

### Inference through Optimization of the likelihood


```python
%%time
m.optimize(maxiter = max_iters)
```

    CPU times: user 7.44 s, sys: 457 ms, total: 7.9 s
    Wall time: 7.55 s





          fun: 20.868585670810997
     hess_inv: <43x43 LbfgsInvHessProduct with dtype=float64>
          jac: array([  8.99958679e-06,   1.41339465e-05,   5.09060783e-05,
             6.72588106e-06,  -1.28315446e-08,   1.22652879e-05,
             5.09060783e-05,   6.72588106e-06,  -1.28315446e-08,
             1.22652879e-05,   5.09060783e-05,   6.72588106e-06,
            -1.28315446e-08,   1.22652879e-05,   5.09060783e-05,
             6.72588106e-06,  -1.28315446e-08,   1.22652879e-05,
             5.09060783e-05,   6.72588106e-06,  -1.28315446e-08,
             1.22652879e-05,   5.09060783e-05,   6.72588106e-06,
            -1.28315446e-08,   1.22652879e-05,   5.09060783e-05,
             6.72588106e-06,  -1.28315446e-08,   1.22652879e-05,
             5.09060783e-05,   6.72588106e-06,  -1.28315446e-08,
             1.22652879e-05,   5.09060783e-05,   6.72588106e-06,
            -1.28315446e-08,   1.22652879e-05,   5.09060783e-05,
             6.72588106e-06,  -1.28315446e-08,   1.22652879e-05,
             5.10663145e-06])
      message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
         nfev: 50
          nit: 42
       status: 0
      success: True
            x: array([  2.1321322 ,  -1.88610378,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.26608891])




```python
def plotprediction(m):
    # Perform prediction
    mu, var = m.predict_f(X_complete)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(labels = df.ds, rotation = 45)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total size of emails in GB');
    ax.plot(X_train.flatten(),Y_train.flatten(), c='b', marker = "o", label = 'Training data')
    ax.plot(X_test.flatten(),Y_test.flatten(), c='r', marker = "o", label = 'Test data')
    ax.plot(X_complete.flatten(), mu.flatten(), c='g', marker = "o", label = "Predicted mean function")
    lower = mu - 2*np.sqrt(var)
    upper = mu + 2*np.sqrt(var)
    ax.plot(X_complete, upper, 'g--', X_complete, lower, 'g--', lw=1.2)
    ax.fill_between(X_complete.flatten(), lower.flatten(), upper.flatten(),
                    color='g', alpha=.1, label = "95% Predicted credible interval")
    plt.legend(loc = "best")
    plt.tight_layout()
```


```python
plotprediction(m);
```


![png](output_27_0.png)



```python
## Calculate the RMSE and MAPE
def calculate_rmse(model, X_test, Y_test):
    mu, var = model.predict_y(X_test)
    rmse = np.sqrt(((mu - Y_test)**2).mean())
    return rmse

def calculate_mape(model, X_test, Y_test):
    mu, var = model.predict_y(X_test)
    mape = (np.absolute(((mu - Y_test)/Y_test)*100)).mean()
    return mape
```


```python
calculate_rmse(model=m, X_test = X_test, Y_test = Y_test)
calculate_mape(model=m, X_test = X_test, Y_test = Y_test)
```




    1.2515806168637664






    27.94536660649003



### Inference through Bayesian Inference


```python
%%time
m.optimize(maxiter = max_iters)
```

    CPU times: user 7.44 s, sys: 457 ms, total: 7.9 s
    Wall time: 7.55 s





          fun: 20.868585670810997
     hess_inv: <43x43 LbfgsInvHessProduct with dtype=float64>
          jac: array([  8.99958679e-06,   1.41339465e-05,   5.09060783e-05,
             6.72588106e-06,  -1.28315446e-08,   1.22652879e-05,
             5.09060783e-05,   6.72588106e-06,  -1.28315446e-08,
             1.22652879e-05,   5.09060783e-05,   6.72588106e-06,
            -1.28315446e-08,   1.22652879e-05,   5.09060783e-05,
             6.72588106e-06,  -1.28315446e-08,   1.22652879e-05,
             5.09060783e-05,   6.72588106e-06,  -1.28315446e-08,
             1.22652879e-05,   5.09060783e-05,   6.72588106e-06,
            -1.28315446e-08,   1.22652879e-05,   5.09060783e-05,
             6.72588106e-06,  -1.28315446e-08,   1.22652879e-05,
             5.09060783e-05,   6.72588106e-06,  -1.28315446e-08,
             1.22652879e-05,   5.09060783e-05,   6.72588106e-06,
            -1.28315446e-08,   1.22652879e-05,   5.09060783e-05,
             6.72588106e-06,  -1.28315446e-08,   1.22652879e-05,
             5.10663145e-06])
      message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
         nfev: 50
          nit: 42
       status: 0
      success: True
            x: array([  2.1321322 ,  -1.88610378,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.63568472,   1.38389724,
            10.77485046,  -1.51730421,   0.26608891])




```python
def plotprediction(m):
    # Perform prediction
    mu, var = m.predict_f(X_complete)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(labels = df.ds, rotation = 45)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total size of emails in GB');
    ax.plot(X_train.flatten(),Y_train.flatten(), c='b', marker = "o", label = 'Training data')
    ax.plot(X_test.flatten(),Y_test.flatten(), c='r', marker = "o", label = 'Test data')
    ax.plot(X_complete.flatten(), mu.flatten(), c='g', marker = "o", label = "Predicted mean function")
    lower = mu - 2*np.sqrt(var)
    upper = mu + 2*np.sqrt(var)
    ax.plot(X_complete, upper, 'g--', X_complete, lower, 'g--', lw=1.2)
    ax.fill_between(X_complete.flatten(), lower.flatten(), upper.flatten(),
                    color='g', alpha=.1, label = "95% Predicted credible interval")
    plt.legend(loc = "best")
    plt.tight_layout()
```


```python
plotprediction(m);
```


![png](output_33_0.png)



```python
## Calculate the RMSE and MAPE
def calculate_rmse(model, X_test, Y_test):
    mu, var = model.predict_y(X_test)
    rmse = np.sqrt(((mu - Y_test)**2).mean())
    return rmse

def calculate_mape(model, X_test, Y_test):
    mu, var = model.predict_y(X_test)
    mape = (np.absolute(((mu - Y_test)/Y_test)*100)).mean()
    return mape
```


```python
calculate_rmse(model=m, X_test = X_test, Y_test = Y_test)
calculate_mape(model=m, X_test = X_test, Y_test = Y_test)
```




    1.2515806168637664






    27.94536660649003



### References
1. Emaasit, D. and Johnson, M. (2018). Capturing Structure Implicitly from Noisy Time-Series having Limited Data. arXiv preprint arXiv:1803.05867.

2. Knudde, N., van der Herten, J., Dhaene, T., & Couckuyt, I. (2017). GPflowOpt: A Bayesian Optimization Library using TensorFlow. arXiv preprint arXiv:1711.03845.

3. Wilson, A. G. (2014). Covariance kernels for fast automatic pattern discovery and extrapolation with gaussian processes. University of Cambridge.

## Computing Environment


```python
# print system information/setup
%reload_ext watermark
%watermark -v -m -p numpy,pandas,gpflowopt,gpflow,tensorflow,matplotlib,ipywidgets,seaborn -g
```

    CPython 3.6.3
    IPython 6.2.1
    
    numpy 1.13.3
    pandas 0.20.3
    gpflowopt 0.1.0
    gpflow 0.4.0
    tensorflow 1.4.1
    matplotlib 2.1.1
    ipywidgets 7.1.1
    seaborn 0.8.1
    
    compiler   : GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)
    system     : Darwin
    release    : 17.3.0
    machine    : x86_64
    processor  : i386
    CPU cores  : 8
    interpreter: 64bit
    Git hash   : 01556a9cc8e59ff71a18726e04c935e3f6ecc585
