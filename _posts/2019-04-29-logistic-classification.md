---
layout: post
title: "Classifying credit defaulters"
subtitle: "A multi-variable logistic model"
author: "Weseem Ahmed "
date: "April 29, 2019"
comments: true
categories: Python
tags: [Regression, classification]
img_url: /includes/20190117-chord-diagram.png
excerpt: This post goes through the steps needed to build a multi-variable logistic model to classify borrowers most likely to default on their loans.
output: html_document 
---

## Intraprovincial Migration
For a study on determining if Ontarians are satsfied with their overall quality of life, one of the ideas my colleagues and 
myself had was to see how people are moving within the province. Following the Dutch people's influx into Amsterdam during the 
Middle Ages as a sign of their approval with their government, the concept of "voting with your feet" has been around for quite 
sometime.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(11, 8))
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
```

Next, we will build a multi-variable logistic model. The model will be trained on the credit card risk dataset. 
The training dataset consists of 300 marked instances and for each instance the information about Credit Card Payment 
`(CC_Payments)` is given together with card owner's wage `(Wage)`, cost of living (Cost_Living), mortgage payment `(Mtg)` and the 
amount the card owner spends on vacations `(Vacations)`. The last column, Default, contains a categorical type value: `Yes` or `No`.

```python
CR = pd.read_csv('/_data/CreditRisk.csv')
Credit_risk = CR[["CC_Payments", 'Wage', 'Cost_Living', 'Mtg','Vacations', 'Default']]
Credit_risk.head()
```

```python
'''Plot the pairwise plot using the seaborn library. 
Notice the parameter 'hue' set to the values in 'Default' column'''

import seaborn as sns

sns.pairplot(Credit_risk, hue='Default')
```
