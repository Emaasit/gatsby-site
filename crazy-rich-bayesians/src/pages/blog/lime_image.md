---
templateKey: article-page
title: Interpretable Machine Learning for Image Classification with LIME
date: 2019-09-23T00:56:55.624Z
cover: "/img/blog_limeimg/banner.png"
tags:
  - MachineLearning
  - Interpretability
  - ImageClassification
  - LIME
meta_title: Interpretable Machine Learning for Image Classification with LIME
meta_description: >-
  How can you evaluate whether the predictions you get from your machine learning model are reliable? LIME is here to the rescue. LIME (Local Interpretable Model-agnostic Explanations) provides explanations for the predictions of any machine learning technique. In this tutorial, we'll see how it works for image classification tasks. 
---
By: Cristian Arteaga, [arteagac.github.io](https://arteagac.github.io)

LIME (Local Interpretable Model-agnostic Explanations) (Ribeiro et. al. 2016) is a popular technique for interpretability in machine learning. LIME provides individual level explanations (one instance in the dataset at the time) for the predictions of any machine learning technique.  

For the case of image clasification, LIME takes as input the image to be explained and a predictive model, and outputs the area of the image with stronger correlation for certain predictions. For instance, let's take as input the following image and a pretrained InceptionV3 model as the predictive model.

![png](/img/blog_limeimg/input_img.png)

When we use the Inception V3 model to predict what is in the image, we get the following top 5 predictions (and their respective probabilities):
- Labrador Retriever (82.2%)
- Golden Retriever (1.5%)
- American Staffordshire Terrier (0.9%)
- Bull Mastiff (0.8%)
- Great Dane  (0.7%)

Now we are going to use LIME to get an explanation of why the machine learning model is predicting that there is a Labrador Retriever in the image.

## LIME Explanations
LIME explanations are created by generating a new dataset of random perturbations around the instance to be explained and then fitting a local surrogate model. This local model is usually a model that is simpler and with intrinsic interpretability such as a linear regression model. For more details about the basics behind LIME I recomend you to check [this tutorial](https://nbviewer.jupyter.org/urls/arteagac.github.io/blog/lime.ipynb).

### Step 1. Create random perturbations of instance being explained
In the context of image classification, perturbations means to compute superpixels in the image being explained and then randomly turn on and off some of the superpixels.
After computing the superpixels in the image we get something like this:
![png](/img/blog_limeimg/superpixels.png)

In order to randomly turn on and off some superpixels we can create a random vector with zeros and ones. Each position in such vector represents whether a superpixel is on (one) or off (zero). Such random vector would look like this: `[1, 0, 0, 1, 0, ...]`. After we generate several of this random vectors we get a **new dataset of perturbations** that can be translated to images like the following:
`perturbation1 = [1, 1, 0, 1, 0, ...]`
![png](/img/blog_limeimg/perturb1.png)
`perturbation2 = [0, 1, 0, 0, 1, ...]`
![png](/img/blog_limeimg/perturb2.png)
`perturbation3 = [1, 0, 0, 1, 1, ...]`
![png](/img/blog_limeimg/perturb3.png)
`perturbation4 = [1, 1, 1, 0, 0, ...]`
![png](/img/blog_limeimg/perturb4.png)
In this example we are showing only a dataset with 4 perturbations. However, in practice, a larger number of perturbations is required for better results. 

### Step 2. Compute predictions for the new generated dataset
The InceptionV3 model is used to predict the class of the new generated images. Remeber that InceptionV3 has 1,000 output classes so the prediction consist of the probability that each image belongs to such 1,000 classes. However, for our example, we will take only the probability for the class "Labrador Retriever". In other words, for each perturbation now we have a probability for the class "Labrador Retriever". For instance:

`P(class labrador|perturbation1) = 0.71`
`P(class labrador|perturbation2) = 0.17`
`P(class labrador|perturbation3) = 0.82`
`P(class labrador|perturbation4) = 0.01`

Therefore, now we have everything to fit a linear model using:
`X = perturbations;   y = predictions for labrador = P(class labrador|perturbations)`

However, before we fit a linear model, LIME needs to give more weight (importance) to images that are closer to the image being explained. This will be done in the next step.

### Step 3. Give more importance to instances in new dataset that are closer to the original image
We can use a distance metric to evaluate how far is each perturbation from the original image. The orinal image is just a perturbation with all the superpixels active (all elements one). Given that the perturbations are multidimensional vectors, the cosine distance is a metric that can be used for this purpose. After the cosine distance has been computed, a kernel function can be used to translate such distance to a value between zero and one (a weight). At the end of this process we have a weight (importance) for each perturbation in the dataset. Something like this:
```
weight1 = kernel_function(distance(perturbation1, original_image))
weight2 = kernel_function(distance(perturbation2, original_image))
weight3 = kernel_function(distance(perturbation3, original_image))
weight4 = kernel_function(distance(perturbation4, original_image))
```

### Step 4. Use perturbations, predictions for labrador and weights to fit a weighed linear model. Use coefficients from the linear model as explanations.
A weighed linear model can now be fitted using the information computed in the previous steps. We get a coefficient for each superpixel in the image. Such coefficient represents how strong is its associated superpixel with the prediction of labrador. We can rank the coefficients with larger magnitude and after rendering their associated superpixels in an image we get something like this:
![png](/img/blog_limeimg/output_img.png)

This is what LIME returns as an explanations. The area of the image (superpixels) that have a stronger effect on the prediction of "Labrador Retriever". A Jupyter Notebook that explains all these steps in detail (with Python code) can be seen [here](https://nbviewer.jupyter.org/url/arteagac.github.io/blog/lime_image.ipynb).

<em>Cristian Arteaga is a Ph.D. Student highly passioned about Machine Learning and Data Science. For more information about Cristian's work check <a href="https://arteagac.github.io" target="_blank" rel="noopener"> his website</a>.</em>

### References
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). Why should i trust you?: Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144). ACM.

