---
layout: post
title: "Classifying credit defaulters"
subtitle: "A machine learning model"
author: "Weseem Ahmed "
date: "April 29, 2019"
comments: true
categories: python
tags: [machine learning]
excerpt: This post builds a machine learning model using a multi-linear logistic regression to estimate which borrowers are more likely to default on their loans.
output: html_document 
---

# Multi-Linear Logistic Regression

Logistic regressions are useful tools for classifying objects. The target variable is binary, either a yes or no, 1 or 0, that we try to estimate. By running the regression on only a subset of data and using the results to estimate it on the rest, we can build a machine learning model. In this example, I'll build a model estimating which consumers are most likely to default on their credit loans from a set of variables.

**Example: Credit Card default risk** 

Another example, which we will consider later in this module, is the prediction of a credit card default event. The dataset contains records that have the default indicator along with card owner wage, the mortgage payment amount, the vacation spending amount, and living cost. The model can be trained on these observations and can be used to predict if default is expected for the new unlabeled instance. Rather, the prediction model is expected to tell how likely the default will occur and thus the card issuer will know the risk. 

In these problems, we require a probability of the event with the positive outcome, it is called 'positive' just because this is the event we are most interested to detect. Ironically, quite often these events are of ill-nature (default or spam message). In the credit card default problem, the goal is to find how likely it is that the default will happen. The target values in the training dataset have two values: either fully paid or charged off. Likewise, in a loan approval, the target values are loan approved or not approved. Analysis of historical data is aimed to find probabilities of positive outcomes (balance fully paid or loan approved), well trained models should be capable to predict what class the new instance belongs to, what makes it a __binary classifier__.  


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(11, 8))
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }


The method `predict` of `sklearn` Logistic Regression predicts the target value, not the probabilities.
For the probabilities, there is another method: `predict_proba`. The last method returned two curves: one shows the probability of 'damage' and the other shows the probability of 'no damage'.

The `sklearn` Logistic Regressor is tunable by __hyperparameter C__ (parameter of the learning algorithm), this parameter controls the 'hardness' of the decision. The larger C is, the harder a classification decision will be to make. 
As an exercise, try to build prediction models with different hyperparameter value C in a range from 1 to 1e9.

## Building our model

Next, we will build a multi-variable logistic model. The model will be trained on the credit card risk dataset. The training dataset consists of 300 marked instances and for each instance the information about Credit Card Payment (`CC_Payments`) is given together with card owner's wage (`Wage`), cost of living (`Cost_Living`), mortgage payment (`Mtg`) and the amount the card owner spends on vacations (`Vacations`). The last column, `Default`, contains a categorical type value: `Yes` or `No`.


```python
CR = pd.read_csv('CreditRisk.csv') # The data is available <a href = "https://github.com/weseemahmed/weseemahmed.github.io/tree/master/_data/CreditRisk.csv"> here </a>

Credit_risk = CR[["CC_Payments", 'Wage', 'Cost_Living', 'Mtg','Vacations', 'Default']]
Credit_risk.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CC_Payments</th>
      <th>Wage</th>
      <th>Cost_Living</th>
      <th>Mtg</th>
      <th>Vacations</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11712</td>
      <td>89925</td>
      <td>44004</td>
      <td>34237</td>
      <td>13236</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4114</td>
      <td>82327</td>
      <td>33111</td>
      <td>35928</td>
      <td>10788</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15941</td>
      <td>68972</td>
      <td>12901</td>
      <td>37791</td>
      <td>8326</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13962</td>
      <td>67582</td>
      <td>21491</td>
      <td>29385</td>
      <td>7647</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6150</td>
      <td>88983</td>
      <td>43084</td>
      <td>29614</td>
      <td>10131</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>

```python
#Plot the pairwise plot using the seaborn library. 
#Notice the parameter 'hue' set to the values in 'Default' column

import seaborn as sns

sns.pairplot(Credit_risk, hue='Default')
```
<p align="center">
  <img alt="credit risk"
  src="{{ site.baseurl }}/img/credit_risk.png"/>
</p>


In the pair plot, the observations labeled as `No` are shown in blue, and `Yes` in orange. Inspection of the graph suggests that observations can be classified by several predictors: `Mtg`, `Vacations`, and `Cost_Living`.

Before building the model, the count of each classes in the dataset should be inspected.


```python
# Count the number of unique values in column "Default".

Credit_risk['Default'].value_counts()
```

### Unbalanced data

For the most accurate results when using logistic regression modeling, the data should be balanced (ie. the number of positive instances should be comparable to the number of negative instances). In the Credit_risk dataframe, there are approximately twice as many observations with negative target value as those with positive. This difference is not too big, so we can proceed with unmodified dataset. 

**NOTE:** Classification algorithms are very sensitive to unbalanced data and for strongly unbalanced datasets, the issue has to be properly addressed before training. The easiest solutions would be: 
* to reduce size of oversampled class or
* to inflate the undersampled class.

To reduce an oversampled class, it is recommended to select that class into a separate dataframe, shuffle the rows and select each n-th row from dataframe; the number n is chosen so that two datasets - positive and negative classes - will have comparible sizes.

To inflate the undersampled class, duplicate the undersampled class so that the final dataset has an approximetely equal number of positive and negative instances. Again, the recipe would be to separate into two datasets by the class, duplicate the rows of the undersampled class as many times as required, and then combine the datasets.

Another approach is to __stratify__ the dataset, that is to fetch randomly selected observations from each subclass and compose a new smaller dataset where both classes are equally represented.


```python
#There is one categorical type in the datasets (column 'Default').
#Convert it to 1 and 0 before training model.

Credit_risk['default_enum'] = Credit_risk['Default'].map({'Yes': 1, 'No': 0})

Credit_risk.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CC_Payments</th>
      <th>Wage</th>
      <th>Cost_Living</th>
      <th>Mtg</th>
      <th>Vacations</th>
      <th>default_enum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>15085.173333</td>
      <td>85327.020000</td>
      <td>35959.893333</td>
      <td>34264.260000</td>
      <td>9789.563333</td>
      <td>0.303333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>24142.424244</td>
      <td>12002.180144</td>
      <td>12550.754681</td>
      <td>15048.593736</td>
      <td>3133.039806</td>
      <td>0.460466</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-40164.000000</td>
      <td>59734.000000</td>
      <td>3654.000000</td>
      <td>10583.000000</td>
      <td>87.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-3234.750000</td>
      <td>76953.750000</td>
      <td>27630.750000</td>
      <td>19554.500000</td>
      <td>7727.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>11422.500000</td>
      <td>84288.000000</td>
      <td>35481.500000</td>
      <td>34198.000000</td>
      <td>10004.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>38863.500000</td>
      <td>93335.750000</td>
      <td>43542.750000</td>
      <td>45641.250000</td>
      <td>11960.250000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>63210.000000</td>
      <td>119703.000000</td>
      <td>75355.000000</td>
      <td>82760.000000</td>
      <td>17421.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

This example has 300 observations, it is not so obvious that observations should be interpreted statistically. 

To illustrate this point, we will split the rows into 14 bins by wage and then calculate the number of observations and the number of defaults in each group. The description of the `Credit_risk` dataset shows wage minimum and maximum values. Based on these values, we'll define the wage bins as:


```python
wage_bins = [55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000]
wage_bin_labels= ['60', '65', '70', '75', '80', '85', '90', '95', '100', '105', '110', '115', '120']

Credit_risk['wage_group'] = pd.cut(Credit_risk.Wage, wage_bins, right=False, labels=wage_bin_labels)
```


```python
#Group the rows by the wage group, count number of rows and defaults in each group.

credit_risk_by_wage= Credit_risk[['default_enum', 'wage_group']].groupby(['wage_group']).agg({'default_enum': 'sum', 'wage_group': 'count'})


#Rename columns
credit_risk_by_wage.columns = ['default', 'Count']

#and calculate the frequency of the default in each group.
credit_risk_by_wage['Proportion'] = credit_risk_by_wage.default / credit_risk_by_wage.Count

credit_risk_by_wage.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wage_group</th>
      <th>default</th>
      <th>Count</th>
      <th>Proportion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65</td>
      <td>2</td>
      <td>5</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70</td>
      <td>10</td>
      <td>27</td>
      <td>0.370370</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75</td>
      <td>16</td>
      <td>27</td>
      <td>0.592593</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80</td>
      <td>18</td>
      <td>45</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>85</td>
      <td>13</td>
      <td>54</td>
      <td>0.240741</td>
    </tr>
    <tr>
      <th>6</th>
      <td>90</td>
      <td>13</td>
      <td>43</td>
      <td>0.302326</td>
    </tr>
    <tr>
      <th>7</th>
      <td>95</td>
      <td>10</td>
      <td>34</td>
      <td>0.294118</td>
    </tr>
    <tr>
      <th>8</th>
      <td>100</td>
      <td>4</td>
      <td>27</td>
      <td>0.148148</td>
    </tr>
    <tr>
      <th>9</th>
      <td>105</td>
      <td>3</td>
      <td>19</td>
      <td>0.157895</td>
    </tr>
    <tr>
      <th>10</th>
      <td>110</td>
      <td>1</td>
      <td>8</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>115</td>
      <td>0</td>
      <td>8</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>120</td>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


```python
plt.figure(figsize=(11, 8))
plt.xlabel('wage group', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Proportion', fontsize=18)
plt.yticks(fontsize=18)

plt.scatter(credit_risk_by_wage.index , credit_risk_by_wage.Proportion)
```
<p align="center">
  <img alt="defaults by wage group"
  src="{{ site.baseurl }}/img/20190429-wage_group_proportion.png"/>
</p>

After grouping by the wage group, the number of instances that fall in each group was counted and the sum of the enumerated default values gives the number of defaults in that group. The calculated proportion gives a rough estimate of the probability of default in the group. We can see that the probability is zero or close to zero for the high wage groups (wage > 105,000) and shows larger value for lower wage group.

This was just an illustration of the **probabilistic** approach to larger datasets. 

The Generalized Linear Model applies a **statistical** model for each observation; the response variable is modeled using a probability distribution and then parameter of the distribution fitted to predictors' values.

As was stated above, the goal of logistic regression is to find the probability that an instance belongs to a positive class. Doing this for logistic regressions is a two-stage process:
* first, the probabilities have to be estimated;
* then, probabilities are fitted to a linear model via a transformation function.

__Note__: `statsmodels` library provides several built-in probability distributions, called _families_: Binomial, Gamma, Gaussian, Negative Binomial, Poisson. The appropriate distribution can be picked if the statistics of the event is known. These distributions will be considered in more detail in the next course of this program. For now, we will use the default distribution: Gaussian, often referred to as **Normal Distribution**. 

Below, let's look at the prediction model for Credit Card default using two predictors - vacation spending (`Vacations`) and mortgage payments (`Mtg`).


```python
plt.figure(figsize=(11, 8))
plt.xlabel('Vacations', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Mtg', fontsize=18)
plt.yticks(fontsize=16)

plt.scatter(Credit_risk.Vacations, Credit_risk.Mtg, c=Credit_risk.default_enum)
```

<p align="center">
  <img alt="mortgages and vacations"
  src="{{ site.baseurl }}/img/20190429-mtg_vac.png"/>
</p>


```python
#Set the Vacations and Mtg as predictors and add constant to predictors for Intercept. 
The target, the value which we are trying to predict is default_enum.

exod_list = Credit_risk[['Vacations', 'Mtg']]
exod_list=sm.add_constant(exod_list)
exod_list

#Train the model
glm_vacations_mtg = sm.GLM(Credit_risk.default_enum, exod_list)

res_multi = glm_vacations_mtg.fit()


#Get the summary of the regression
res_multi.summary()
```




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>default_enum</td>   <th>  No. Observations:  </th>  <td>   300</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   297</td>  
</tr>
<tr>
  <th>Model Family:</th>       <td>Gaussian</td>     <th>  Df Model:          </th>  <td>     2</td>  
</tr>
<tr>
  <th>Link Function:</th>      <td>identity</td>     <th>  Scale:             </th> <td> 0.10310</td> 
</tr>
<tr>
  <th>Method:</th>               <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -83.365</td> 
</tr>
<tr>
  <th>Date:</th>           <td>Sat, 05 Jan 2019</td> <th>  Deviance:          </th> <td>  30.621</td> 
</tr>
<tr>
  <th>Time:</th>               <td>20:15:29</td>     <th>  Pearson chi2:      </th>  <td>  30.6</td>  
</tr>
<tr>
  <th>No. Iterations:</th>         <td>3</td>        <th>  Covariance Type:   </th> <td>nonrobust</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>     <td>   -0.5388</td> <td>    0.075</td> <td>   -7.157</td> <td> 0.000</td> <td>   -0.686</td> <td>   -0.391</td>
</tr>
<tr>
  <th>Vacations</th> <td> 9.082e-06</td> <td> 5.93e-06</td> <td>    1.531</td> <td> 0.126</td> <td>-2.54e-06</td> <td> 2.07e-05</td>
</tr>
<tr>
  <th>Mtg</th>       <td> 2.198e-05</td> <td> 1.23e-06</td> <td>   17.806</td> <td> 0.000</td> <td> 1.96e-05</td> <td> 2.44e-05</td>
</tr>
</table>




```python
#To get parameters of the fit

res_multi.params
```




    const       -0.538818
    Vacations    0.000009
    Mtg          0.000022
    dtype: float64




```python
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 1], Credit_risk.Mtg[Credit_risk.default_enum == 1], label = 'No Default')
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 0], Credit_risk.Mtg[Credit_risk.default_enum == 0], label = 'Default')
plt.plot(Credit_risk.Vacations, (0.5388/0.000009) - (0.00002198/0.000009082)* Credit_risk.Vacations, 'r',label='Decision Boundary')
plt.legend()
```

<p align="center">
  <img alt="mortgages and vacations"
  src="{{ site.baseurl }}/img/20190429-default_fit_model.png"/>
</p>


```python
#Set the Vacations and Mtg as predictors and add constant to predictors for Intercept. 
The target is default_enum.

X_cr = Credit_risk[['Vacations', 'Mtg']]
Y_cr = Credit_risk['default_enum'].values

from sklearn.linear_model import LogisticRegression

#Instantiate  the model
log_reg = LogisticRegression(C=1e10)

#Train the model
log_reg.fit(X_cr, Y_cr)
```

    LogisticRegression(C=10000000000.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)




```python
#Predict the targets

Y_log_reg_pred = log_reg.predict(X_cr)
```


```python
#To estimate goodness of the fit, we evalute the accuracy score
#which counts the number of correctly predicted observations.

from sklearn.metrics import accuracy_score

accuracy_score(Y_cr, Y_log_reg_pred)
```
    0.75



The linear models relies on a number of factors:
* the linear dependence between the **predictors** and the **target**;
* **independence** of the observations in the dataset

The linear models are robust, easy to implement and to interpret. However, if listed assumptions do not hold, these models would not produce accurate predictions and more complex methods are required.
    


----

## References

Connor Johnson blog (http://connor-johnson.com/2014/02/18/linear-regression-with-python/).

MNIST database (https://en.wikipedia.org/wiki/MNIST_database).

Ng, A. (2018)  _Machine Learning Yearning_ (electronic book).

Unsupervised Learning. (https://en.wikipedia.org/wiki/Unsupervised_learning#Approaches).

Witten, I.H, Frank, E. (2005) *Data Mining. Practical Machine Learnng Tools and Techniques* (2nd edition). Elsevier.


```python

```
