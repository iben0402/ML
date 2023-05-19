import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def calculate_hypothesis(theta, X):
    return np.dot(X, theta)

def calculate_cost(theta, X, y):
    predictions = calculate_hypothesis(theta, X)
    cost = (1 / 2) * np.sum(np.square(predictions - y))
    return cost


plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

# datasets
df = pd.read_csv('insurance.csv')

# preparing data
categorical_columns = ["sex", "children", "smoker", "region"]
df_encode = pd.get_dummies(df, columns=categorical_columns, prefix='OHE', prefix_sep='_', drop_first=True, dtype='int8')


print(df_encode.head())
X = df_encode.drop('charges', axis=1)  # Independent variable
y = df_encode['charges']  # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)
# added x0 = 1 to dataset
X_train_0 = np.c_[np.ones((X_train.shape[0],1)),X_train]
X_test_0 = np.c_[np.ones((X_test.shape[0],1)),X_test]

theta = np.matmul(np.linalg.inv(np.matmul(X_train_0.T, X_train_0)), np.matmul(X_train_0.T, y_train))

# print(theta)
# print(calculate_cost(theta, X_train_0, y_train))

parameter = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
columns = ['intersect:x_0=1'] + list(X.columns.values)
parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

sk_theta = [lin_reg.intercept_] + list(lin_reg.coef_)
parameter_df = parameter_df.join(pd.Series(sk_theta, name='sk_theta'))
print(parameter_df)

plt.plot(theta, sk_theta)
plt.show()