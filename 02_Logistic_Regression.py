import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

basetable = pd.read_csv("datasets/basetable.csv")

logreg = LogisticRegression()
X = basetable[["age"]]
y = basetable[["target"]]
logreg.fit(X,y)

print(logreg.coef_)

print(logreg.intercept_)

## Multivariate logistic Regression

X = basetable[["gender_F","age","time_since_last_gift"]]
y = basetable[["target"]]
logreg.fit(X,y)
print(logreg.coef_)
print(logreg.intercept_)
## Make prediction
logreg.predict_proba([[1,72,120]])








