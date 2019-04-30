
# Module 9: Introduction to Machine Learning Part 2

This module continues introduction to Machine Learning. It covers Logistic Regression, Support Vector Machines, Decision Trees and Random Forest algorithms.

## Logistic Regression

In the previous module, we considered the linear regression model, when both the target and the predictors are numerical values. 

In this module, we will continue working with linear models and introduce __logistic regression__. 

In many machine learning tasks, the goal is to classify objects. To perform such classification, the algorithms need to find the probability that an instance belongs to one class or another. 

**Example: Spam filters**

An automated spam filter estimates the probability that an incoming message is spam. The computed probability is compared to a pre-set threshold probability. In this case, 50% is a natural choice. If a harder filter is required the threshold can be set to a lower value, say 20%, but a harder filter would send more messages to the spam folder. 

The model is trained on the labeled datasets, where each observation is labeled as 'spam' or 'ham' (ham is a label for not spam), and described by a number of features - the number of recipients, the count of specific words ("winner", "fantastic offer", "password") in the subject line and in the body of the message, and other criteria to classify the message.

**Example: Credit Card default risk** 

Another example, which we will consider later in this module, is the prediction of a credit card default event. The dataset contains records that have the default indicator along with card owner wage, the mortgage payment amount, the vacation spending amount, and living cost. The model can be trained on these observations and can be used to predict if default is expected for the new unlabeled instance. Rather, the prediction model is expected to tell how likely the default will occur and thus the card issuer will know the risk. 

In these problems, we require a probability of the event with the positive outcome, it is called 'positive' just because this is the event we are most interested to detect. Ironically, quite often these events are of ill-nature (default or spam message). In the credit card default problem, the goal is to find how likely it is that the default will happen. The target values in the training dataset have two values: either fully paid or charged off. Likewise, in a loan approval, the target values are loan approved or not approved. Analysis of historical data is aimed to find probabilities of positive outcomes (balance fully paid or loan approved), well trained models should be capable to predict what class the new instance belongs to, what makes it a __binary classifier__.  

In a classification task, the type of target value is **categorical** - spam or ham, default or no default, loan fully paid or charged off, approved or disapproved. Since everything is a number, we will have 1 for 'yes' and 0 for 'no', or +1 for 'yes' and -1 for 'no'; categorical type values must be enumerated for machine learning analysis. The training set observations represent two classes: positive (with label 'yes') and negative (with label 'no'). 

Generally speaking, regression techniques can be applied to a classification problem. Regression can be performed for each class from a training set independently by setting the response to one for instances of class with positive outcomes and zero for other instances. The result will be the linear expression for the class. To predict the unknown instance, the value of both linear expressions is calculated, then the largest value points to the class the new instance belongs to. This approach is called _multiresponse linear regression_.

A multiresponse linear regression often produces good results. However, in most cases the data can be analyzed only statistically, and the multiresponse method is not suitable for such analysis. The calculated values are not the probabilities of belonging to a certain class (values can fall outside the probability range of 0 and 1).

Although most datasets do not show sharp separation into two classes, it is still possible to find the likelihood of a given observation belonging to a positive class. In fact, this makes the logistic regression very useful. The example below illustrates the probabilistic approach in classification problems.

### Logit method. The Challenger example

The first dataset we will study is based on the Challenger space shuttle disaster of 1986, when one of the rocket boosters exploded. After the data was analyzed, the commission determined that the explosion was caused by the failure of an O-ring in the rocket booster which was unacceptably sensitive to the outside temperature and other factors. The dataset contains data from 24 flights prior to the disaster.

The dataset home page can be found [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Challenger+USA+Space+Shuttle+O-Ring).
You can also read about the event in [Wikipedia](https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster).


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(11, 8))
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

challenger_pd = pd.read_csv("challenger_data.csv")
challenger_pd['Intercept'] = np.ones((len(challenger_pd),))
challenger_pd.columns = ['Date', 'Temperature', 'DamageIncident', 'Intercept']

challenger_pd
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
      <th>Date</th>
      <th>Temperature</th>
      <th>DamageIncident</th>
      <th>Intercept</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04/12/1981</td>
      <td>66</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11/12/1981</td>
      <td>70</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3/22/82</td>
      <td>69</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6/27/82</td>
      <td>80</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01/11/1982</td>
      <td>68</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>04/04/1983</td>
      <td>67</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6/18/83</td>
      <td>72</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8/30/83</td>
      <td>73</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11/28/83</td>
      <td>70</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>02/03/1984</td>
      <td>57</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>04/06/1984</td>
      <td>63</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8/30/84</td>
      <td>70</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10/05/1984</td>
      <td>78</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>11/08/1984</td>
      <td>67</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1/24/85</td>
      <td>53</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>04/12/1985</td>
      <td>67</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4/29/85</td>
      <td>75</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6/17/85</td>
      <td>70</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7/29/85</td>
      <td>81</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>8/27/85</td>
      <td>76</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10/03/1985</td>
      <td>79</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10/30/85</td>
      <td>75</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>11/26/85</td>
      <td>76</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>01/12/1986</td>
      <td>58</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1/28/86</td>
      <td>31</td>
      <td>Challenger Accident</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''The last record can be removed from the dataset as
that record is not part of training dataset.'''

challenger_24 = challenger_pd.drop([24])

'''Missing values should be removed too.'''

challenger = challenger_24.dropna()

plt.figure(figsize=(11, 8))
plt.xlabel('Temperature, F', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Damage Incident', fontsize=18)
plt.yticks(fontsize=18)
plt.scatter(challenger["Temperature"], challenger["DamageIncident"], alpha=0.95)
```




    <matplotlib.collections.PathCollection at 0x1117f1eb8>




![png](output_8_1.png)


This scatter plot shows that at temperatures higher than 76 F no failure was observed, all starts at the temperatures below 65 F showed O-ring failure and at the intermediate temperatures, O-ring failure happened some of the time. In other words, at each temperature there is  a _probability_ that failure will occur. While at high temperatures that probability is zero or close to zero, at low temperatures the probability of damage gets very close to 1. The graph shows that the probability of damage incidents increases as the outside temperature decreases.

Next, let's build a Linear Regression predictive model:
1. instantiate the model
2. fit the model to observed data


```python
'''Linear regressor is imported from sklearn package'''

from sklearn.linear_model import LinearRegression

'''Instantiate  the model'''
lin_reg = LinearRegression()

'''Prepare the predictor X and the target Y.'''
X_cd_lr = challenger['Temperature'].values.reshape(-1,1)
Y_cd_lr = challenger['DamageIncident']

'''Fit the model.'''
lin_reg.fit(X_cd_lr, Y_cd_lr)

'''Print the coefficients of the fit'''
print(lin_reg.coef_, lin_reg.intercept_)

plt.figure(figsize=(11, 8))
plt.xlabel('Temperature, F', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Damage Incident', fontsize=18)
plt.yticks(fontsize=18)

plt.scatter(challenger['Temperature'], challenger['DamageIncident'], color="b", alpha=0.5)
plt.plot(X_cd_lr, lin_reg.predict(X_cd_lr))
```

    [-0.03738095] 2.90476190476
    




    [<matplotlib.lines.Line2D at 0x113dac898>]




![png](output_10_2.png)


Even though the fitted line shows the trend correctly, the fit does not look good. Most data points fall far from the fitted line, at high temperatures the fitted values are less than 0. 

Most importantly, it is not clear how to use this fit for finding probabilities and predicting the new instances. 

The probability of the damage should be described with a function that gradually changes as temperature rises and is bound between 0 and 1. 

The logistic function - also called **logit** - is the most popular choice for logistic regression. This function has the form $$f(x) = \frac{1}{1+e^{-ax}}$$ and is demonstrated below.


```python
'''Definition of the logistic function for demo.'''

def logistic_function(x, alpha):
    return 1./(1 + np.exp(-alpha*(x)))

x = np.linspace(-100, 100, 1000)

plt.figure(figsize=(11, 8))
plt.xlabel('x', fontsize = 18)
plt.xticks(fontsize=18)
plt.ylabel('Logit (x)', fontsize = 18)
plt.yticks(fontsize=18)

'''Plot of the logit function with positive and negative coeffitient a.'''
plt.plot(x, logistic_function(x, 0.10), label=r"a = 0.10")
plt.plot(x, logistic_function(x, -0.10), label=r"a = -0.10")
plt.legend()
```




    <matplotlib.legend.Legend at 0x113fa90b8>




![png](output_12_1.png)


The `logit` function has values bound between 0 and 1 and gradually increases/decreases for positive/negative parameter 'a'.

The `Logit` method is available from the `statsmodels` library and there is no need to code the `logit` function explicitly.

Let's try this method with our Challenger problem:



```python
import statsmodels.api as sm

'''Set Temperature and Intercept as predictors, 
and the target is DamageIncident column.'''

X_logit = challenger[['Temperature', 'Intercept']].values
Y_logit = challenger['DamageIncident'].values.astype(np.float)


'''Perform the fit with Logit method.'''

logitReg = sm.Logit(Y_logit, X_logit)
logit_fit = logitReg.fit()
```

    Optimization terminated successfully.
             Current function value: 0.441635
             Iterations 7
    


```python
'''Inspect the summary of the fit.'''

logit_fit.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>y</td>        <th>  No. Observations:  </th>  <td>    23</td> 
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>    21</td> 
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td> 
</tr>
<tr>
  <th>Date:</th>          <td>Sat, 05 Jan 2019</td> <th>  Pseudo R-squ.:     </th>  <td>0.2813</td> 
</tr>
<tr>
  <th>Time:</th>              <td>19:29:12</td>     <th>  Log-Likelihood:    </th> <td> -10.158</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -14.134</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>0.004804</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th>    <td>   -0.2322</td> <td>    0.108</td> <td>   -2.145</td> <td> 0.032</td> <td>   -0.444</td> <td>   -0.020</td>
</tr>
<tr>
  <th>const</th> <td>   15.0429</td> <td>    7.379</td> <td>    2.039</td> <td> 0.041</td> <td>    0.581</td> <td>   29.505</td>
</tr>
</table>



### Prediction using the `Logit` method

Now that we have a trained model, we can use it for the prediction of new instances. Temperature on the day of the Challenger disaster was 31 F. We can now use the model and the Python `predict` method. Since our predictor X was set as a two-dimensional vector (Temperature), and constant "1", a list with two values [31, 1] will be passed as an argument.


```python
logit_fit.predict([31, 1])
```




    array([ 0.99960878])



Our model predicts that the probability of failure at that temperature is 99.96% and so we can predict the failure of the O-ring with high probability.

As was stated before, logistic regression estimates the probability that an instance belongs to a _positive class_, (a class with target values equal to one). To make a prediction - or classify a new object - a **threshold** is required. Often, the threshold is set to 50%, so that the prediction whether an instance belong to class (the expected response value y = 1) or not (y = 0) can be made by comparing the estimated probability to the threshold:

$$y = 1 \verb+ if + p < 0.5$$
$$y = 0 \verb+ if + p \geq 0.5$$

This suggests existence of a **decision boundary** that separates positive outcomes from negative ones. The boundary is drawn from the threshold value or rather at the predictor value X such that the probability is 0.5. Next, we shall find the location of the decision boundary in our Challenger example.


```python
'''This will find the value of X at which probability is 0.5.
To get a more accurate answer, we generate 1000 points in the range [1, 100].'''

x_array = np.linspace(1, 100, 1000)
ones = np.ones(1000)
X_cd_pred = np.column_stack([x_array, ones])

'''And predict the response for these X values.'''
Y_cd_pred = logit_fit.predict(X_cd_pred)

'''Finally,  the last point with probability larger 
or equal 0.5 is the decision boundary.'''
X_cd_pred[Y_cd_pred[:] >= 0.5][-1]
```




    array([ 64.72072072,   1.        ])




```python
'''The result can be predicted using 'predict' method.'''
Y_logit_pred = logit_fit.predict(X_logit)

plt.figure(figsize=(11, 8))
plt.xlabel('Temperature, F', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Damage Incident', fontsize=18)
plt.yticks(fontsize=18)
plt.scatter(challenger['Temperature'], challenger['DamageIncident'], color="b", alpha=0.5, label='O-ring failure')

plt.scatter(X_logit[:,0], logit_fit.predict(X_logit), color="r")
plt.legend(('O-ring failure', 'Logit fit'), loc='upper right', fontsize= 16)
plt.plot([64.72, 64.72], [0,1], 'k:', color='g')
```




    [<matplotlib.lines.Line2D at 0x115034898>]




![png](output_21_1.png)


### The logistic regression formula and cost function

The graph above shows observations (blue dots), the `Logit` regression predictions (red circles) and the decision boundary at T = 64.72 (green dotted line).

Linear regression attempts to model the relationship between the target and predictor by fitting a linear equation $$y = w_0 + \sum_{i=1}^L w_i x_i$$ to observed data. 

Logistic regression is also the linear model, the observed data is fitted by expression of the same type but this time the linear expression (below)

$$w_0 + \sum_{i=1}^L w_i x_i$$ is wrapped into a transformation function, like this:

$$transformation(p_i) = w_0 + \sum_{i=1}^L w_i x_i$$

For the one-variable case, linear expression takes the simple form: $w_0 + w_1 x_1$ and when `Logit(p)` is used as the transformation function:

$$Logit(p_i) = w_0 + w_{1} x_{1}$$
or 

$$p_i = \frac{1}{1+e^{w_0 + w_1 x_1}}$$

The probability p<sub>i</sub> equals 0.5 when w<sub>0</sub> + w<sub>1</sub> x<sub>1</sub> = 0. The last equation defines decision boundary: x<sub>boundary</sub> = - w<sub>0</sub>/w<sub>1</sub>.

For multi-variable problems:
$$Logit(p_i) = w_0 + \sum_{i=1}^L w_i x_i$$

and the probabilities

$$p_i = \frac{1}{1+e^{w_0 + \sum_{i=1}^L w_{1} x_{1}}}$$
The decision boundary is defined by linear equation:

$$w_0 + \sum_{i=1}^L w_i x_i = 0$$

Again, like in the linear regression the solution, the list of parameters w<sub>0</sub>, w<sub>1</sub>, ..., w<sub>L</sub> - is found by **minimizing the cost function**. 

The cost function has to be of special form. The model has to estimate high probabilities for positive instances (y = 1) and low probabilities for negative instances (y = 0). To satisfy these conditions, the following function is utilized:
 
 
  \begin{cases} -log(p)\verb+ if +y = 1 \\ -log(1-p)\verb+ if +y = 0 \end{cases}
  for a single instance.
  
For a whole training set, the average cost function over all instances is taken:

  $$-\frac{1}{m_i}\sum_{i=1}^L [y^i log(p^i) + (1 - y^i) log(1 - p^i)]$$

Instead of log(p<sup>i</sup>) and log(1 - p<sup>i</sup>) the corresponding linear expressions are substituted and the problem is solved by finding coefficients w<sub>i</sub> to minimize the cost function.  


```python
'''The decision boundary can be found from the parameters of our model. 
params[0] corresponds to the intercept W_0
and params[1] to the slope W_1 so that equation W_0 + W_1 * X_boundary = 0 has a simple solution (-W_0/W_1).'''

X_boundary = (-1) * logit_fit.params[1]/logit_fit.params[0]
X_boundary
```




    64.794640924551445



### `scikit-learn` Logistic Regression

The _Logistic Regressor_ is also part of `scikit-learn` Python library. For demonstration we shall apply it to the Challenge disaster data.

'''scikit-learn library also has Logistic Regression as part of linear-model block.'''

from sklearn.linear_model import LogisticRegression

'''Instantiate the model'''


log_reg_challenger = LogisticRegression(C=1e9)

X_cd_logreg = challenger['Temperature'].values.reshape(-1,1)
Y_cd_logreg = challenger['DamageIncident'].values

'''fit the model to oberved data'''

log_reg_challenger.fit(X_cd_logreg, Y_cd_logreg)


```python
X_pred = np.linspace(50, 85, 1000).reshape(-1,1)
plt.scatter(challenger["Temperature"], challenger["DamageIncident"])

'''And predict the response for X_pred values.'''

Y_pred= log_reg_challenger.predict(X_pred)
plt.plot(X_pred, Y_pred)
plt.xlabel('Temperature, F', fontsize=18)
plt.ylabel('Damage Incident', fontsize=18)

plt.plot(X_pred, log_reg_challenger.predict_proba(X_pred))
```




    [<matplotlib.lines.Line2D at 0x1151d1da0>,
     <matplotlib.lines.Line2D at 0x1151d1eb8>]




![png](output_28_1.png)


The method `predict` of `sklearn` Logistic Regression predicts the target value, not the probabilities.
For the probabilities, there is another method: `predict_proba`. The last method returned two curves: one shows the probability of 'damage' and the other shows the probability of 'no damage'.

The `sklearn` Logistic Regressor is tunable by __hyperparameter C__ (parameter of the learning algorithm), this parameter controls the 'hardness' of the decision. The larger C is, the harder a classification decision will be to make. 
As an exercise, try to build prediction models with different hyperparameter value C in a range from 1 to 1e9.

## Linear models and multi-variable logistic regression

Next, we will build a multi-variable logistic model. The model will be trained on the credit card risk dataset. The training dataset consists of 300 marked instances and for each instance the information about Credit Card Payment (`CC_Payments`) is given together with card owner's wage (`Wage`), cost of living (`Cost_Living`), mortgage payment (`Mtg`) and the amount the card owner spends on vacations (`Vacations`). The last column, `Default`, contains a categorical type value: `Yes` or `No`.


```python
CR = pd.read_csv('CreditRisk.csv')
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
'''Plot the pairwise plot using the seaborn library. 
Notice the parameter 'hue' set to the values in 'Default' column'''

import seaborn as sns

sns.pairplot(Credit_risk, hue='Default')
```




    <seaborn.axisgrid.PairGrid at 0x113fd64a8>




![png](output_33_1.png)


In the pair plot, the observations labeled as `No` are shown in blue, and `Yes` in orange. Inspection of the graph suggests that observations can be classified by several predictors: `Mtg`, `Vacations`, and `Cost_Living`.

Before building the model, the count of each classes in the dataset should be inspected.


```python
# '''Count the number of unique values in column "Default".'''

Credit_risk['Default'].value_counts()
```

### Unbalanced data

For best results using logistic regression modeling, the data should be balanced, meaning the number of positive instances should be comparable to the number of negative instances. In the Credit_risk dataframe, there are approximately twice as many observations with negative target value as those with positive. This difference is not too big, so we can proceed with unmodified dataset. 

**NOTE:** Classification algorithms are very sensitive to unbalanced data and for strongly unbalanced datasets, the issue has to be properly addressed before training. The easiest solutions would be: 
* to reduce size of oversampled class or
* to inflate the undersampled class.

To reduce an oversampled class, it is recommended to select that class into a separate dataframe, shuffle the rows and select each n-th row from dataframe; number n is chosen so that two datasets - positive and negative classes - will have comparible sizes.

To inflate the undersampled class, duplicate the undersampled class so that the final dataset has an approximetely equal number of positive and negative instances. Again, the recipe would be to separate into two datasets by the class, duplicate the rows of the undersampled class as many times as required, and then combine the datasets.

Another approach is to __stratify__ the dataset, that is to fetch randomly selected observations from each subclass and compose a new smaller dataset where both classes are equally represented.


```python
'''There is one categorical type in the datasets (column 'Default').
Convert it to 1 and 0 before training model.'''

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



In the Challenger example, there are only 24 observations and the corresponding graph demonstrates the statistical meaning of the observations. In the larger datasets, for example Credit Card dataset with 300 observations, it is not so obvious that observations should be interpreted statistically. 

To illustrate this point, we will split the rows into 14 bins by wage and then calculate the number of observations and the number of defaults in each group. The description of the `Credit_risk` dataset shows wage minimum and maximum values. Based on these values, the wage bins are defined as:


```python
wage_bins = [55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000]
wage_bin_labels= ['60', '65', '70', '75', '80', '85', '90', '95', '100', '105', '110', '115', '120']


Credit_risk['wage_group'] = pd.cut(Credit_risk.Wage, wage_bins, right=False, labels=wage_bin_labels)
```


```python
'''Group the rows by the wage group, count number of rows and defaults in each group.'''

credit_risk_by_wage= Credit_risk[['default_enum', 'wage_group']].groupby(['wage_group']).agg({'default_enum': 'sum', 'wage_group': 'count'})


'''Rename columns'''
credit_risk_by_wage.columns = ['default', 'Count']

'''and calculate the frequency of the default in each group.'''
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




    <matplotlib.collections.PathCollection at 0x116745278>




![png](output_42_1.png)


After grouping by the wage group, the number of instances that fall in each group was counted and the sum of the enumerated default values gives the number of defaults in that group. The calculated proportion gives a rough estimate of the probability of default in the group. We can see that the probability is zero or close to zero for the high wage groups (wage > 105,000) and shows larger value for lower wage group.

This was just an illustration of the **probabilistic** approach to larger datasets. 

The Generalized Linear Model applies a **statistical** model for each observation; the response variable is modeled using a probability distribution and then parameter of the distribution fitted to predictors' values.

As was stated above, the goal of logistic regression is to find the probability that an instance belongs to a positive class. In the Challenger example, it was obvious that at any given temperature, there is a certain probability of an O-ring failure. The point is that logistic regression is a two-stage process:
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




    <matplotlib.collections.PathCollection at 0x116917ba8>




![png](output_45_1.png)



```python
'''Set the Vacations and Mtg as predictors and add constant to predictors for Intercept. 
The target, the value which we are trying to predict is default_enum.'''

exod_list = Credit_risk[['Vacations', 'Mtg']]
exod_list=sm.add_constant(exod_list)
exod_list

'''Train the model'''
glm_vacations_mtg = sm.GLM(Credit_risk.default_enum, exod_list)

res_multi = glm_vacations_mtg.fit()


'''Get the summary of the regression'''
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
'''To get parameters of the fit'''

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




    <matplotlib.legend.Legend at 0x11696a748>




![png](output_48_1.png)



```python
'''Set the Vacations and Mtg as predictors and add constant to predictors for Intercept. 
The target is default_enum.'''

X_cr = Credit_risk[['Vacations', 'Mtg']]
Y_cr = Credit_risk['default_enum'].values

from sklearn.linear_model import LogisticRegression

'''Instantiate  the model'''
log_reg = LogisticRegression(C=1e10)

'''Train the model'''
log_reg.fit(X_cr, Y_cr)
```




    LogisticRegression(C=10000000000.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)




```python
'''Predict the targets'''

Y_log_reg_pred = log_reg.predict(X_cr)
```


```python
'''To estimate goodness of the fit we evalute the accuracy score,
which counts the number of correctly predicted observations.'''

from sklearn.metrics import accuracy_score

accuracy_score(Y_cr, Y_log_reg_pred)
```




    0.75



The linear models relies on a number of factors:
* the linear dependence between the **predictors** and the **target**;
* **independence** of the observations in the dataset

The linear models are robust, easy to implement and to interpret. However, if listed assumptions do not hold, these models would not produce accurate predictions and more complex methods are required.
    

## Support Vector Machines (SVM)

A Support Vector Machines (SVM) is another family of machine learning models. SVM algorithms are very powerful and versatile, they can be applied to both classification and regression problems, and are used often for outlier detection. They are not limited to linear problems, and are capable of performing non-linear classification and regression. 

SVM classifiers approach classification differently. Recall that logistic regression computes the probability that an instance belongs to a positive class, and makes placement decisions comparing value of that probability with a given threshold value (for example, 50%). The instance is then placed into a positive or negative class based on this decision. The decision boundary is a 'side result'. 

An SVM Classifier is focused on the separating plane itself. Using the training dataset, it tries to find the plane that best separates instances labeled as positive from instances labeled as negative. Let's take for example the datasets with two predictors x<sub>1</sub> and x<sub>2</sub> and the response y<sub>i</sub>. The separating plane (a line in this case) is defined as: 

$$w_0 + w_1 x_1 + w_2 x_2 = 0$$ or $$x_2 = (-w_0/w_2) + (-w_1/w_2) x_1$$

The other way to formulate it would be to say that the decision function $$w_0 + w_1 x_1 + w_2 x_2$$ returns zero for any x<sub>1</sub> and x<sub>2</sub> that fall on the decision line. For all other points, it would give a positive value for all points on one side of the plane and negative for the other side. This makes the SVM prediction very easy, the value of the decision function for an instance with predictor values x<sub>1</sub> and x<sub>2</sub> has to be computed. The SVM classifier prediction can be formulated as:

$$y =\begin{cases} 0\verb+ if +w_0 + w_1 x_1 + w_2 x_2 < 0, \\ 1\verb+ if +w_0 + w_1 x_1 + w_2 x_2 >= 0 \end{cases}$$

### Training the SVM model

To train an SVM classifier model means to find the coefficients w<sub>i</sub> so that the decision boundary $w_0 + w_1 x_1 + w_2 x_2 = 0$ best separates two classes. The corresponding algorithm tries to find the boundary that stays as far away from the instances as possible. Each instance of the training dataset is assigned a target value depending on the class: 

\begin{cases} t_i = -1\verb+ if +y_i = 0 \verb+ (negative instances),+ \\ t_i = +1\verb+ if +y_i = 1 \verb+ (positive instances) +\end{cases}

The proper decision function would give a positive value for positive instances ($w_0 + w_1 x_1 + w_2 x_2 > 0$), and a negative value for negative instances. Actually, the decision function should return a value either larger than +1 or smaller than -1. Thus the product $t_i (w_0 + w_1 x_1 + w_2 x_2)$ should be always positive if the optimal decision function was found. For data sets that cannot be sharply separated, the objective is to find the decision function that minimizes violations, when the product of the target and the decision function value is negative.    

Let's try this with the credit risk data set:


```python
from sklearn.svm import SVC
```


```python
svm_clf = SVC(kernel = 'linear', C=1)
svm_result = svm_clf.fit(X_cr, Y_cr)
```


```python
'''Extract the coefficients '''
coef = svm_result.coef_

Intercept = svm_result.intercept_[0]

margin = 1/coef[0][1]
margin

coef
```




    array([[ 0.00246713,  0.05781414]])




```python
plt.figure(figsize=(12,7))
plt.subplot(121)
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 1], Credit_risk.Mtg[Credit_risk.default_enum == 1], label = 'Default')
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 0], Credit_risk.Mtg[Credit_risk.default_enum == 0], label = 'No Default')
plt.plot(Credit_risk.Vacations, (0.5388/0.000009) - (0.00002198/0.000009082)* Credit_risk.Vacations, 'r',label='Decision Boundary')
plt.legend()


plt.subplot(122)
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 1], Credit_risk.Mtg[Credit_risk.default_enum == 1], label = 'Default')
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 0], Credit_risk.Mtg[Credit_risk.default_enum == 0], label = 'No Default')
plt.plot(Credit_risk.Vacations, (-Intercept/coef[0][1]) - (coef[0][0]/coef[0][1])* Credit_risk.Vacations, 'r',label='Decision Boundary')
plt.plot(Credit_risk.Vacations, (-Intercept/coef[0][1]) + margin - (coef[0][0]/coef[0][1])* Credit_risk.Vacations, 'r--',label='Decision Boundary')
plt.plot(Credit_risk.Vacations, (-Intercept/coef[0][1]) - margin - (coef[0][0]/coef[0][1])* Credit_risk.Vacations, 'r--',label='Decision Boundary')

#plt.ylim(43000, 43800)
plt.legend()
margin
```




    17.296805656982091




![png](output_60_1.png)



```python
accuracy_score(Y_cr, svm_clf.predict(X_cr))
```




    0.86333333333333329



As you can see above, the SVM classifier found a different separation line from that produced by logistic regression. While it is not possible to separate instances ideally in this dataset, the decision boundary is drawn so that there is approximately the same number of violations on both sides. Indeed, the algorithm placed the boundary as far as possible from these instances.

Just like the logistic regressor, the SVM classifier is tunable by hyperparameter C. In the example above, the hyperparameter was set to 100. For smaller value of C = 1, the result would be a bit different.


```python
from sklearn.svm import LinearSVC
svm_softer = SVC(C=0.01, kernel='linear')
svm_softer_result = svm_softer.fit(X_cr, Y_cr)

coef_softer = svm_softer_result.coef_

Intercept_softer = svm_softer_result.intercept_[0]


margin_softer = 1/coef_softer[0][1]
margin_softer

```




    1394.9164257741675




```python
plt.figure(figsize=(12,7))
plt.subplot(121)
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 1], Credit_risk.Mtg[Credit_risk.default_enum == 1], label = 'Default')
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 0], Credit_risk.Mtg[Credit_risk.default_enum == 0], label = 'No Default')
plt.plot(Credit_risk.Vacations, (-Intercept/coef[0][1]) - (coef[0][0]/coef[0][1])* Credit_risk.Vacations, 'r',label='Decision Boundary')
plt.plot(Credit_risk.Vacations, 
         (-Intercept/coef[0][1]) + margin - (coef[0][0]/coef[0][1])* Credit_risk.Vacations, 'r--',label='Decision Boundary')
plt.plot(Credit_risk.Vacations, (-Intercept/coef[0][1]) - margin - (coef[0][0]/coef[0][1])* Credit_risk.Vacations, 'r--',label='Decision Boundary')
plt.ylim(37500, 47500)

plt.subplot(122)
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 1], Credit_risk.Mtg[Credit_risk.default_enum == 1], label = 'Default')
plt.scatter(Credit_risk.Vacations[Credit_risk.default_enum == 0], Credit_risk.Mtg[Credit_risk.default_enum == 0], label = 'No Default')
plt.plot(Credit_risk.Vacations, (-Intercept_softer/coef_softer[0][1]) - (coef_softer[0][0]/coef_softer[0][1])* Credit_risk.Vacations, 'r',label='Decision Boundary')
plt.plot(Credit_risk.Vacations, 
         (-Intercept_softer/coef_softer[0][1]) - margin_softer -(coef_softer[0][0]/coef_softer[0][1])* Credit_risk.Vacations, 
         'r--',label='Decision Boundary')
plt.plot(Credit_risk.Vacations, 
         (-Intercept_softer/coef_softer[0][1]) + margin_softer -(coef_softer[0][0]/coef_softer[0][1])* Credit_risk.Vacations, 
         'r--',label='Decision Boundary')
plt.ylim(37500, 47500)


```




    (37500, 47500)




![png](output_64_1.png)


The result of the two classifiers above show different decision lines and a different margin, which was found as $1/w_1. The instances located closest to the decision boundary fully determines the boundary. More precisely, the instances that are located in between margins define the result; these instances are called the __support vectors__.  

When hyperparameter C is set to a large value, the model will return a result with a small margin value, and enforce all instances outside of margins. This is called _hard margin classification_. On the left side, the margins are very small.
**NOTE:** The hard margin classifier is not really recommended for the cases when data points are not clearly separable, as it might give large prediction error.

At low C, the margins are larger and instances are allowed in the region between margins. This is called - soft margin classifier. The softer classifier is actually expected to make less prediction error.  

Support vector machines are also capable of performing _regression_. As in the classification algorithm, the algorithm is based on the instances located close to the margins. But this time, the training objective is reversed. If a classifier tries to reduce margin violations and squeeze the instances outside, the SVM Regressor tries to include as many points as possible into the space between the margin. The width of the space between margins is controlled by a hyperparameter `epsilon`. The larger `epsilon` produces wider space. Example of instantiation of SVM linear regressor `LinearSVR` is shown below.


```python
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=5)
```

## Training vs Test datasets

When you start working with the dataset, it is always recommended to split your original dataset into two datasets, **training** dataset and **test** dataset. 

The **training** subset is used to train the model and fit the model parameters.
The **test** subset is used to test how well your model performs on the new data.

Why do we need to split the data into two subsets? Let's imagine that we created a model and fine-tuned the parameters so well that the accuracy of the model is almost 100% and it perfectly fits the data which we have. What will be the accuracy of the model on a new data? Will the model work for the data which was not used to train the model? The accuracy of the model can only be determined by considering how well a model performs on new data that was not used when fitting the model.

Below are two examples when the model is **overfitted**, meaning that the model performs very well on the training data but it does not generalize well. Both examples are used in [this Wikipedia article](https://en.wikipedia.org/wiki/Overfitting) on overfitting.

**Example 1:** The green line represents an overfitted classification model and the black line represents a better generalized model. While the green line perfectly describes the training data, it is likely to have a high error rate on new data, compared to the black line.

<img src='https://upload.wikimedia.org/wikipedia/commons/1/19/Overfitting.svg' height="300" width="300" alt="Overfitted classification model (green line)">

**Image source:** Image by [Chabacano - Own work, CC BY-SA 4.0](https://commons.wikimedia.org/w/index.php?curid=3610704)

**Example 2:** In this example, noisy data is perfectly fitted to a polynomial function (blue line). Even though the polynomial function provides a perfect fit, it won't perform well on a new data. The simple linear function would be a better fit for this data.

<img src='https://upload.wikimedia.org/wikipedia/commons/6/68/Overfitted_Data.png' height="400" width="400" alt="Overfitted model">

**Image source:** Image by [Ghiles - Own work, CC BY-SA 4.0](https://commons.wikimedia.org/w/index.php?curid=47471056)


Usually, the original dataset is split 70/30 or 80/20. This means that 70% (or 80%) of all data points are used to  train the model and the remaining data is reserved for testing.

Each model trained on the train subset can be tested on the test subset to see its predictive performance. This allows fine tuning of the prediction model.  

## Decision Trees

Decision Trees is a special family of Machine Learning algorithms. These algorithms are versatile and can perform both classification and regression tasks. They are very powerful, can be used to fit complex datasets but also they are used as a component of **Random Forests** algorithms, which are among the most powerful Machine Learning algorithms available today.

Below, the `DecisionTreeClassifier` is applied to the Credit Card dataset. The workflow is the same as for the logistic regression or SVM classifier:
1. instantiate the model
2. fit the model to observed data (fitting returns a Results class)


```python
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_cr, Y_cr)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
'''The tree can be visualized with export_graphviz'''

from sklearn.tree import export_graphviz


dot_data = export_graphviz(tree_clf, 
                 out_file=None,
               feature_names=Credit_risk.columns[4:2:-1],
               class_names=Credit_risk.Default.unique(),
               rounded=True,
               filled=True)
```


```python
import graphviz

graph = graphviz.Source(dot_data)
graph
```




![svg](output_75_0.svg)




```python
accuracy_score(Y_cr, tree_clf.predict(X_cr))
```




    0.88



Let's climb the decision tree that we have grown to see how the new instance can be classified.
* The topmost node, called the __root node__ asks if the value of mortgage (`Mtg`) is less than or equal to `45224.5`. If the answer is `yes`, move down along the left (`True`) branch to the left child node; if the answer is `no` (`False`), move down the tree to the right child node. These two nodes are at the `depth = 1` level, each node divides the data further. 

Each node has several attributes:
* A `samples` attribute counts the number of instances it was applied to. The root node is applied to the whole dataset, then `221` samples have a `Mtg` value of less than or equal to `45224.5` and the data is further divided into two groups of `135` and `86` samples;
*  A `value` attribute shows the number of instances for each class in this node; the top node includes `300` samples, out of which `209` instances are in the negative class (`No`), and `91` instances are in the positive class (`Yes`).
* The node `gini` attribute gives the measure of the node impurity. If `gini = 0`, the node is pure and all instances in the given node belong to the same class. In the example above, there is one pure node, when `Mtg <= 31064`. 

No default was observed in the training dataseton the when `Mtg <= 31064` side. On the other side, if `Mtg` value is larger than `50718.5`, the default is very likely, out of `50` samples placed in this group `48` will result in default.
    
To train Decision Trees the Scikit-learn uses the **Classification And Regression Tree (CART)** algorithm. First, the algorithm splits the whole training set in two subsets on single feature `i` and threshold t<sub>i</sub>. During this step, the algorithm searches for the `i` and t<sub>i</sub> that gives the purest subsets with smallest `gini` indices. Then, each subset is split again using the same logic and so on. This recursive split stops when algorithm cannot further reduce impurity or the maximum depth is reached.


```python
'''Let's build deeper tree with 3 depth levels'''

tree_clf_deeper = DecisionTreeClassifier(max_depth=3)
tree_clf_deeper.fit(X_cr, Y_cr)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
dot_data_deeper = export_graphviz(tree_clf_deeper, 
                 out_file=None,
               feature_names=Credit_risk.columns[4:2:-1],
               class_names=Credit_risk.Default.unique(),
               rounded=True,
               filled=True)

graph_depth_3 = graphviz.Source(dot_data_deeper)
graph_depth_3
```




![svg](output_79_0.svg)




```python
accuracy_score(Y_cr, tree_clf_deeper.predict(X_cr))
```




    0.89666666666666661



With maximum_depth = 3 the better splits were found and accuracy_score slightly improved. With larger number of depth the accuracy may be improved, but actually the model may be __overfit__. The fitting accuracy may be very high, but prediction of the new instances might be poor.
That is why, as we discussed before, it is highly recommended to split the dataset into train and test subsets and check prediction on the data the model has not seen (test subset).  

## Decision Trees - regression model

As it was mentioned above, Decision Trees can also perform regression tasks. Let's apply `DecisionTreeRegressor` to the `bike_sharing` dataset used in the previous module.


```python
bike_sharing = pd.read_csv('bikes_sharing.csv', header = 0, sep = ',')
bike_sharing.head(5)
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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
bike_sharing.datetime = bike_sharing.datetime.apply(pd.to_datetime)
bike_sharing['month'] = bike_sharing.datetime.apply(lambda x : x.month)
bike_sharing['hour'] = bike_sharing.datetime.apply(lambda x: x.hour)
bike_sharing_15 = bike_sharing.loc[bike_sharing['hour'] == 15]
bike_sharing_15.head(5)
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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>month</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>2011-01-01 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>18.04</td>
      <td>21.970</td>
      <td>77</td>
      <td>19.9995</td>
      <td>40</td>
      <td>70</td>
      <td>110</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2011-01-02 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>13.94</td>
      <td>16.665</td>
      <td>81</td>
      <td>11.0014</td>
      <td>19</td>
      <td>55</td>
      <td>74</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2011-01-03 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.120</td>
      <td>30</td>
      <td>16.9979</td>
      <td>14</td>
      <td>58</td>
      <td>72</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2011-01-04 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>11.48</td>
      <td>13.635</td>
      <td>52</td>
      <td>16.9979</td>
      <td>17</td>
      <td>48</td>
      <td>65</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2011-01-05 15:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>12.30</td>
      <td>14.395</td>
      <td>28</td>
      <td>12.9980</td>
      <td>7</td>
      <td>55</td>
      <td>62</td>
      <td>1</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
import sklearn

Y = bike_sharing_15["count"]
X = bike_sharing_15["temp"].values.reshape(-1, 1)

# the model to do the fit
model = sklearn.linear_model.LinearRegression().fit(X, Y)
print(model.coef_)
print(model.intercept_)
```

    [ 6.99568077]
    91.4256037508
    


```python
from sklearn.tree import DecisionTreeRegressor


tree_reg_depth_2 = DecisionTreeRegressor(max_depth=2)
model_tree_reg_depth_2 = tree_reg_depth_2.fit(X, Y)

x_bike_pred = np.linspace(5.0, 40.0, 500).reshape(-1,1)
```


```python
plt.figure(figsize=(12,7))

plt.subplot(121)
plt.scatter(x=bike_sharing_15["atemp"], y=bike_sharing_15["count"])
plt.plot(X, model.predict(X), 'r-')
plt.xlabel("temp")
plt.ylabel('Count')

plt.subplot(122)
plt.scatter(x=bike_sharing_15["atemp"], y=bike_sharing_15["count"])
plt.plot(x_bike_pred, model_tree_reg_depth_2.predict(x_bike_pred), 'r-')
```




    [<matplotlib.lines.Line2D at 0x119cb6ac8>]




![png](output_88_1.png)


Here, the results of the LinearRegressor and DecisionTreeRegressor are compared. The result of the last method looks very different, DecisionTreeRegressor built a tree (shown below). This tree is very similar to the classification tree, but this time in each node it predicts a value, not a class. The **mse** (mean square error) might look high, but remember that to evaluate the LinearRegressor results we used RMSE - root mean square error. So, to make errors comparable, the square root of MSE has to be calculated.

At the bottom of the decision tree four _leaf nodes_ are located; these nodes predict a single (average) value of response for the range of predictor values. The predicted bike count values is:  
* 94.182 when `temp` <= 11.89, 
* 175.313 when `temp` value is larger than 11.89 but smaller or equal 15.99,
* and so on.  

These values are represented on the graph, the predicted value for each region is the average target value of the instances in that region. 


```python
reg_result_depth_2 = export_graphviz(tree_reg_depth_2, 
                 out_file=None,
               feature_names=["temp"],
               rounded=True,
               filled=True)

graph_reg_depth_2 = graphviz.Source(reg_result_depth_2)
graph_reg_depth_2
```




![svg](output_90_0.svg)



Next, the models with more depth levels are trained on the `bike_sharing_15` dataset. The results are presented on the graphs. 


```python
'''the deeper trees with depth 4 and 6'''

tree_reg_depth_4 = DecisionTreeRegressor(max_depth=4)
model_tree_reg_depth_4 = tree_reg_depth_4.fit(X, Y)

tree_reg_depth_6 = DecisionTreeRegressor(max_depth=6)
model_tree_reg_depth_6 = tree_reg_depth_6.fit(X, Y)


```


```python
plt.figure(figsize=(16,8))

plt.subplot(131)
plt.scatter(x=bike_sharing_15["atemp"], y=bike_sharing_15["count"])
plt.plot(x_bike_pred, model_tree_reg_depth_2.predict(x_bike_pred), 'r-')

plt.subplot(132)
plt.scatter(x=bike_sharing_15["atemp"], y=bike_sharing_15["count"])
plt.plot(x_bike_pred, model_tree_reg_depth_4.predict(x_bike_pred), 'r-')

plt.subplot(133)
plt.scatter(x=bike_sharing_15["atemp"], y=bike_sharing_15["count"])
plt.plot(x_bike_pred, model_tree_reg_depth_6.predict(x_bike_pred), 'r-')
```




    [<matplotlib.lines.Line2D at 0x119dd5c50>]




![png](output_93_1.png)


## Ensemble Learning

In the subsections above, we described several methods (Linear Regressor and Classifier, Support Vector Machine and Decision Tree). We tried to tune each predictor to improve quality. Working with different datasets, you may find that for a given dataset one of the known predictors might give better results than another. And for a different dataset, another predictor is the champion. Or even worse, none of the predictors give acceptable prediction accuracy. If this is the case, it would be nice to aggregate the predictions of a group of predictors. The aggregate often produces better predictions than even the best predictor of the group. Even if few good predictors were built, the accuracy might still be improved by combining them into an __ensemble__. 

In the next cell, predictions of logistic regression, support vector machine and decision tree classifiers are aggregated with Scikit-Learn `VotingClassifier`. The `voting` parameter was set to `hard`, __hard voting__ means that the class that gets most votes is used for majority rule voting. If all classifiers in an aggregate can estimate class probabilities (have `predict_proba` method), the most likely class can be predicted. This is the case of __soft voting__, the `voting` parameter can be set to `soft` to predict the class with highest probability.


```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('lr', log_reg), ('svc', svm_clf), ('dtree', tree_clf)], voting='hard')
voting_clf.fit(X_cr, Y_cr)
```




    VotingClassifier(estimators=[('lr', LogisticRegression(C=10000000000.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)), ('sv...      min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'))],
             flatten_transform=None, n_jobs=1, voting='hard', weights=None)




```python
Y_voting_pred = voting_clf.predict(X_cr)
accuracy_score(Y_cr, Y_voting_pred)
```




    0.8666666666666667



### Bagging and Pasting. Random Forest

Above to compose an ensemble the different algorithms were applied to the same dataset. Another approach is to apply the same training algorithm to different random subsets of the dataset. 
There are two methods to draw observations from training dataset:
 * when observations are drawn from dataset with replacement the method is called __bagging__,
 * without replacement - __pasting__.

Each predictor of an ensemble is trained on the portion of training dataset. Then, ensemble makes a prediction by aggregating the predictions. The average value is taken from regressors and the class with most votes in classification problems. 

`BaggingClassifier` and `BaggingRegressor` are available from `sklearn.ensemble` package. Number of classifiers and number of training instances can be set as parameters `n_estimator` and `n_samples`, respectively.


```python
from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, max_samples=200, bootstrap=True)

bagging_clf.fit(X_cr, Y_cr)

Y_bagging_pred = bagging_clf.predict(X_cr)

accuracy_score(Y_cr, Y_bagging_pred)
```




    0.97333333333333338



A __Random Forest__ is an ensemble of Decision Trees, it works by training many Decision Trees just like `BaggingClassifier` above. While any available Classifier or Regressor can be used to build Bagging Classifier/Regressor, example above demonstrates bagging with Decision Trees. However, the Random Forest is more optimized for Decision Trees. Here, the `RandomForestClassifier` is used to build predictive model for Credit Card problem.


```python
from sklearn.ensemble import RandomForestClassifier

'''Instantiate the Random Forest Classifier'''
rf_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=32)


'''Train the model'''
rf_clf.fit(X_cr, Y_cr)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=32,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
'''Predict'''
Y_rf_pred = rf_clf.predict(X_cr)

'''Evaluate the prediction accuracy'''
accuracy_score(Y_cr, Y_rf_pred)
```




    0.98999999999999999



The `RandomForestRegressor` can be controlled by the number of regressors using `n_estimators`, and the depth level of each tree can be limited by setting parameter `max_depth`.

-----------

You have reached the end of this module. 

----

## References

Connor Johnson blog (http://connor-johnson.com/2014/02/18/linear-regression-with-python/).

MNIST database (https://en.wikipedia.org/wiki/MNIST_database).

Ng, A. (2018)  _Machine Learning Yearning_ (electronic book).

Unsupervised Learning. (https://en.wikipedia.org/wiki/Unsupervised_learning#Approaches).

Witten, I.H, Frank, E. (2005) *Data Mining. Practical Machine Learnng Tools and Techniques* (2nd edition). Elsevier.


```python

```
