# Project 2 - Blood donation

## Context

You will work on the [Blood donation](https://www.drivendata.org/competitions/2/) dataset as presented in the datadriven.org [competition](https://www.drivendata.org/competitions/2/)
The dataset is from a mobile blood donation vehicle in Taiwan. The Blood Transfusion Service Center drives to different universities and collects blood as part of a blood drive. We want to predict whether or not a donor will give blood the next time the vehicle comes to campus.
Read the [data explanation](https://www.drivendata.org/competitions/2/page/7/) on the datadriven.org website. You may have to create an account to access these pages.
The [competition forum](https://community.drivendata.org/c/warm-up-predict-blood-donations) may have some insights on the data.

The assignment consists of the following elements.

1. A thorough graphic exploration and statistical analysis of the data
2. Comparing several linear regression models and putting forward arguments on their vailidity and reliability
3. Predicting blood donation on a test set. We will not use the competition's logloss metric but instead the raw prediction from your best linear model.


## Part I : Explore the training dataset

* import with read_csv('filename', index=id) so you load the id column from the csv file as your dataframe index
* You can rename the columns to something easier to manipulate if you want

### Data Exploration
* Graphic exploration of the dataset: density plots, scatter plots, box plots. Use matplotlib or seaborn
* Look at the Correlation matrix of the predictors and target. What can you infer?
* In this dataset does correlation imply causation? Are the conditions for causality respected?
What potential confounders can you think of?

### Data Munging
* Check for outliers, missing values
* Check for multicolinearity. If you detect multicollinearity, what action would you take?
* Is the test data distribution similar to the train data?
* Are the features normaly distributed?
    * Use a [QQ plot](https://en.wikipedia.org/wiki/Q–Q_plot) to check for normality. See also [here](http://stackoverflow.com/questions/13865596/quantile-quantile-plot-using-scipy) or [here](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.probplot.html)
    * What type of correction can you apply? Does it improve the results?

### Linear modeling
* Define a multiple regression model with 'Made Donation in March 2007' as the response variable and the other predictor
variables. State the model assumptions.

* Interpret the summary of the least squares fit obtained
* Is the fitted model, as a whole, significant (at least one explanatory variable is useful)?
* Define the null and alternative hypotheses about the regression coefficients, report the value
of the test statistic and its P-value from the output, and give your conclusion.

* try some feature engineering:
    * Transform your predictors, (log, square, combining columns, ...), remove or add new columns
    * Report the ideas you've tried even if they don't improve the results

* Conclusion: Does it look that a linear model is appropriate for describing the relationship between donations and these predictors? Why or why not?

## Part II : Prediction

* Use your best model to predict blood donations on the test dataset
* Check that the results are coherent (visually or statistically)
* Save the results in a csv file according to the BloodDonationSubmissionFormat.csv file. male sure the index / id are correct

## Submission

* Title your notebook : Project 02 Firstname Lastname and make sure the file is named: Project-02-Firstname-Lastname.ipynb
* Clone the repo git@github.com:alexperrier/ga-students.git
* cd ga-students
* Add your file to the folder unit-projects
* Commit, push
    * if you have an error when you push it's probably because someone already pushed a modification. In that case do a *git pull* and then retry to git push
* Check your work is on github https://github.com/alexperrier/ga-students/tree/master/unit-projects

The due date for project 2 is Lesson 8 (Thursday 7/14).

