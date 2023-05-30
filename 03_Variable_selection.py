import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

basetable = pd.read_csv("datasets/basetable.csv")

## Multivariate logistic Regression
logreg = LogisticRegression()
X = basetable[["gender_F","age","time_since_last_gift"]]
y = basetable[["target"]]
logreg.fit(X,y)
print(logreg.coef_)
print(logreg.intercept_)
y_pred = logreg.predict_proba(X)[:,1]

## Model evaluation: AUC

import numpy as np
from sklearn.metrics import roc_auc_score
roc_auc_score(y,y_pred)

## Forward Stepwise Var. Selection

### Calculating AUC func

def auc(variables, target, basetable):
  X = basetable[variables]
  y = basetable[target]
  logreg = LogisticRegression()
  logreg.fit(X,y)
  predictions = logreg.predict_proba(X)[:,1]
  auc = roc_auc_score(y, predictions)
  return float(auc)


auc = auc(["age","gender_F"],["target"],basetable)
print(round(auc,2))

### Calculating the next best variable
def next_best(current_variables,candidate_variables, target, basetable):
    best_auc = -1
    best_variable = None
    for v in candidate_variables:
        auc_v = auc(current_variables + [v], target,basetable)
        if auc_v >= best_auc:
            best_auc = auc_v
            best_variable = v
    return best_variable

current_variables = ["age","gender_F"]
candidate_variables = ["min_gift","max_gift","mean_gift"]
next_variable = next_best(current_variables,candidate_variables,["target"],basetable)
print(next_variable)


### The forward Selection by 5

candidate_variables = ['gender_F',
 'income_high',
 'income_low',
 'country_USA',
 'country_India',
 'country_UK',
 'age',
 'time_since_last_gift',
 'time_since_first_gift',
 'max_gift',
 'min_gift',
 'mean_gift']

current_variables = []
target = ["target"]
max_number_variables = 5
number_iterations = min(max_number_variables,len(candidate_variables))
for i in range(number_iterations):
    next_var = next_best(current_variables,candidate_variables,target,basetable)
    current_variables = current_variables + [next_var]
    candidate_variables.remove(next_var)


print(current_variables)


