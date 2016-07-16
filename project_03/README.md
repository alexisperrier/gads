# Project 3 - Classification

In this project you will work on the Caravan Insurance Dataset and apply all the methods and models we've seen so far to predict whether individuals bought Insurance or not.

The Caravan Dataset includes 85 predictors that measure demographic characteristics for 5822 individuals.
As you will see the characteristics are not explicit. The field names go by denominations such as: MOSTYPE  MAANTHUI MGEMOMV MGEMLEEF and we don't know what these means.

The response variable is **Purchase**, a binary variable. Only 348 persons have bought the Caravan Insurance out of 5822 individuals.

The dataset is available in the [https://github.com/alexperrier/gads/](https://github.com/alexperrier/gads/tree/master/project_03/data) repo

### 1. data exploration

Get familiar with the predictors, understand their distribution, correlations, ... but don't spend too much time looking into each predictor. You want to have a global understanding of the data.

* What's the scale of the features?
* Are there some features that contain mostly zeros?

### 2. test train split

Shuffle your data and split the dataset into a training (80) and a testing (20) set. Don't forget to make your analysis reproducible by setting a seed before the split.

* Take care of transforming the Purchase variable into a binary (0/1).

### 3. Classify!

Using K-Fold Cross validation, (K to your choosing), train the following models and compare their performance.

* Logistic Regression
* KNN
* LDA
* QDA

For each model:

* Use GridSearchCV when applicable to tune the hyper parameters of your models (K-NN for instance).
* Analyse the confusion matrix to understand the performance of your model
* Plot the ROC curve and calculate the AUC

Looking at the different metrics defined around the confusion matrix, what is the best metric to capture your models performances? (Sensitivity, Recall, Accuracy, .... others?

### 4. Features

You noticed that some features are more informative than others (mainly composed of zeros for instance). While others are more correlated with one another.

Try the following techniques and see if that improves your models performances

1) Try removing the features that have less "signal".


2) Use PCA to project the feature space into a lower space and at the same time whiten (decorrelate) the features.

For both techniques do you see any improvement? Is it the same for all models?




