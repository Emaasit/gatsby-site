---
templateKey: article-page
title: Principled AI with Probabilistic Machine Learning
date: 2018-05-22T19:29:55.624Z
cover: "/img/bayes-theorem-neon-lights.jpg"
tags:
  - PrincipledAI
  - MachineLearning
meta_title: Principled AI with Probabilistic Machine Learninga
meta_description: >-
  Last month, I gave a presentation titled *Introduction to Probabilistic Machine Learning using PyMC3* at two local meetup groups (Bayesian Data Science D.C. and Data Science &amp; Cybersecurity) in McLean, Virginia. The following is a summary of the concepts we discussed regarding **Principled AI**.
---
**(Note: Cross-posted with the [Haystax Technology Blog](https://haystax.com/blog/).)**

At Haystax Technology, we are proponents and early adopters of principled approaches to machine learning (ML) and artificial intelligence (AI) for cybersecurity.

We use the term 'principled AI' to describe what we call our model-based approach, which is built on coherent mathematical principles. These principles help us keep our AI transparent, explainable and interpretable. Most importantly, they enable our systems to quantify uncertainty, unlike the black-box approach of deep neural networks. Our users and followers often hear us evangelize this principled approach through publications and conferences, boot camps and local meetups.

Last month, I gave a presentation titled "*Introduction to Probabilistic Machine Learning using PyMC3*" at two local meetup groups (Bayesian Data Science D.C. and Data Science &amp; Cybersecurity) in McLean, Virginia:

<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">Guests just devoured delicious kabobs and spirits and now our <a href="https://twitter.com/Emaasit?ref_src=twsrc%5Etfw">@Emaasit</a> is showing us how to build probabilistic models in computer code using PyMC3 (<a href="https://t.co/kWxaUL0ZGM">https://t.co/kWxaUL0ZGM</a>) a Python PP language.  Join future Data Science Meetups: <a href="https://t.co/vvHaXwT6ms">https://t.co/vvHaXwT6ms</a> <a href="https://t.co/9tDTKW95sp">pic.twitter.com/9tDTKW95sp</a></p>&mdash; Haystax Technology (@HaystaxTech) <a href="https://twitter.com/HaystaxTech/status/989649743818223622?ref_src=twsrc%5Etfw">April 26, 2018</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


The following is a summary of the concepts we discussed during the meetup.

### General Overview

Many data-driven solutions in cybersecurity are seeing a heavy use of machine learning to detect and predict cyber crimes. This may include monitoring streams of network data and predicting unusual events that deviate from the norm. For example, an employee downloading large volumes of intellectual property (IP) on a weekend. **Immediately**, we are faced with our first challenge, that is, we are dealing with quantities (unusual volume &amp; unusual period) whose values are uncertain. To be more concrete, we start off very uncertain whether this download event is unusually large and then slowly get more and more certain as we uncover more clues such as the period of the week, performance reviews for the employee, did they visit WikiLeaks, etc.

In fact, the need to deal with uncertainty arises throughout our increasingly data-driven world. Whether it is Uber autonomous vehicles dealing with predicting pedestrians on roadways or Amazon's logistics apparatus that has to optimize its supply chain system. All these applications have to handle and manipulate uncertainty. Consequently, we need a principled framework for quantifying uncertainty which will allow us to create applications and build solutions in ways that can represent and process uncertain values. Fortunately, there is a simple framework for manipulating uncertain quantities which uses probability to quantify the degree of uncertainty. To quote Prof. Zhoubin Ghahramani, Uber's Chief Scientist and Professor of AI at University of Cambridge:
<blockquote>Just as Calculus is the fundamental mathematical principle for calculating rates of change, Probability is the fundamental mathematical principle for quantifying uncertainty.</blockquote>
This has resulted in a principled approach to machine learning based on probability theory called <strong>Probabilistic Machine Learning(PML)</strong>. It is an exciting area of research that is currently receiving a lot of attention in many conferences (<a href="https://nips.cc/">NIPS</a>, <a href="http://www.auai.org/">UAI</a>, <a href="https://www.aistats.org/">AISTATS</a>), journals (<a href="http://www.jmlr.org/">JMLR</a>, <a href="https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34">IEEE PAMI</a>), open-source software tools (<a href="https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245">TensorFlow Probability</a>, <a href="https://eng.uber.com/pyro/">Pyro</a>) and practical applications at notable companies such as <a href="http://uber.ai/">Uber AI</a>, <a href="https://research.fb.com/prophet-forecasting-at-scale/">Facebook AI Research</a>, <a href="https://x.company/loon/">Google AI</a>, <a href="https://www.microsoft.com/en-us/research/project/infernet/">Microsoft Research</a>.
<h3>Probabilistic Machine Learning</h3>
In general, Probabilistic Machine Learning can be defined as an interdisciplinary field focusing on both the mathematical foundations and practical applications of systems that learn <strong>models</strong> from data. It brings together ideas from Statistics, Computer Science, Engineering and Cognitive Science as illustrated in the Figure below.

<img class="wp-image-6502 size-medium" src="https://haystax.com/wp-content/uploads/2018/05/Screen_Shot_2018-04-25_at_1.17.31_PM-352x282.png" alt="" width="352" height="282" /> Image Credit: http://mlg.eng.cam.ac.uk/zoubin/

In this framework, a **model** is defined as a description of data one could observe from a system. In other words, a model is a set of assumptions that describe the process by which the observed data was generated. This model can be developed graphically inform of a Probabilistic Graphical Model (PGM) as illustrated in the Figure below.

<img class="aligncenter wp-image-6503" src="https://haystax.com/wp-content/uploads/2018/05/Screen-Shot-2018-05-21-at-3.19.02-PM-1024x491.png" alt="" width="600" height="288" />

In the Figure above, the circular nodes represent random variables for the uncertain quantities (e.g. unusual volume or unusual period) and the square nodes represent the uncertainty over the corresponding quantities (e.g. probability of unusual volume). The downward arrow shows the direction of the process that generated the data. The upward arrow shows the direction of inference, that is, given observed data we can learn the parameters of the probability distributions that generated the observed data. As we observe more and more data, our uncertainty over the random variables (e.g. unusual volume) decreases. This is the modern view of machine learning according to Prof. Chris Bishop of Microsoft Research.

Learning follows from two simple rules of probability, namely:
<ul>
 	<li>The sum rule: $p(\mathbf{\theta}) = \sum_{y} p(\mathbf{\theta}, y)$</li>
 	<li>The product rule: $p(\mathbf{\theta}, y) = p(\mathbf{\theta}) p(y \mid \mathbf{\theta})$</li>
</ul>
These two rules can be formulated into Bayes Theorem which tells us the new information we have gained about our original hypothesis (or parameters) given observed data.

\begin{equation}\label{eqn:bayes}
p(\mathbf{\theta}\mid \textbf{y}) = \frac{p(\textbf{y} \mid \mathbf{\theta}) \, p(\mathbf{\theta})}{\textbf{y}},
\end{equation}

where:

\begin{aligned}
p(\mathbf{\theta}\mid \textbf{y}) &= \text{the posterior distribution of the hypothesis (or parameters), given the observed data} \\
p(\textbf{y} \mid \mathbf{\theta}) &= \text{the data likelihood, given the hypothesis (or parameters)} \\
p(\mathbf{\theta}) &= \text{the prior over all possible hypotheses (or parameters)} \\
p(\textbf{y}) &= \text{the data (constant)}
\end{aligned}  


The probabilistic approach to machine learning has proven to be preferable to deep learning in many applications that require transparency and oversight. Although deep learning has produced amazing performance on many benchmark tasks in specific applications such as computer vision and conversational AI (e.g in the recent <a href="https://youtu.be/D5VN56jQMWM">Google Duplex</a>), it has several limitations in much more general and broader use cases such as Cybersecurity, and <a href="https://blogs.wsj.com/cio/2018/05/11/bank-of-america-confronts-ais-black-box-with-fraud-detection-effort/">Banking</a>. Deep learning systems are generally:

<ul>
 	<li type="square">very <strong>data hungry</strong> (i.e. often require millions of examples for training)</li>
 	<li type="square">very <strong>compute-intensive</strong> to train and deploy (i.e. require cloud GPU &amp; TPU resources)</li>
 	<li type="square">poor at representing <strong>uncertainty</strong></li>
 	<li type="square"><strong>easily fooled</strong> by adversarial examples</li>
 	<li type="square"><strong>finicky to optimize</strong>: choice of architecture, learning procedure, etc, require expert knowledge and experimentation</li>
 	<li type="square">uninterpretable <strong>black-boxes</strong>, lacking in transparency, difficult to trust</li>
</ul>

In contrast, PML systems are transparent, explainable, do not require lots of data and computer power. Currently, it is easier than ever to get started building PML systems. This is attributed to the plethora of open source software tools called Probabilistic Programming Languages. These include Google's TensorFlow Probability, Uber's Pyro, Microsoft's Infer.Net, PyMC3, Stan, and many others.

These are a few of the topics that we discussed during this meetup. Materials from the meetup including slides and source code are provided below.

<iframe style="border: 3px solid #EEE;" src="//slides.com/emaasit/intro-pml-dc/embed" width="576" height="420" frameborder="0" scrolling="no" allowfullscreen="allowfullscreen"></iframe>

<em>Daniel Emaasit is a Data Scientist at Haystax Technology. For a more detailed treatment of this subject, please see <a href="https://www.danielemaasit.com/post/2017/08/14/introduction-to-probabilistic-machine-learning/" target="_blank" rel="noopener">Daniel's blog</a>.</em>

#### Source code
For interested readers, two options are provided below to access the source code used for the demo:

1. The entire project (code, notebooks, data and results) can be found <a href="https://github.com/Emaasit/meetups/tree/master/2018_04_26_Intro_to_PML_DC" target="_blank" rel="noopener">here</a> on GitHub.

2. Click the Binder icon below to open the notebooks in a web browser and explore the entire project without downloading and installing any software.[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Emaasit/meetups/master?urlpath=lab) 

#### References
1. Ghahramani, Z. (2015). Probabilistic machine learning and artificial intelligence. Nature, 521(7553), 452.

2. Bishop, C. M. (2013). Model-based machine learning. Phil. Trans. R. Soc. A, 371(1984), 20120222.

3. Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT Press.

4. Barber, D. (2012). Bayesian reasoning and machine learning. Cambridge University Press.

5. Salvatier, J., Wiecki, T. V., &amp; Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. PeerJ Computer Science, 2, e55.
