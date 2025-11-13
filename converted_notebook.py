import pandas as pd
import numpy as np




house = pd.read_csv("Bengaluru_House_Data.csv")
house.head()

house.area_type.unique()

len(house.availability.unique())

len(house.location.unique())

len(house.society.unique())

house.drop(["availability",'location','society'], axis=1, inplace=True)

house.head()

house.dropna(inplace=True)

house['size'].unique()
house['bhk'] = house['size'].apply(lambda x: x.split(" ")[0]).astype(int)
house.head()

house.drop("size", axis=1, inplace=True)

house.head()

house = house[(house.total_sqft.str.isnumeric())]

house.total_sqft = house.total_sqft.astype(float)

house.info()

house_encoded = pd.get_dummies(house, dtype=int)

house_encoded.head()

X = house_encoded.drop('price', axis=1)
y = house_encoded['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4545)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# training
model = LinearRegression()
model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)
print(f"Training score: {r2_score(y_train,y_pred_train)}")
print(f"Absolute error: {mean_absolute_error(y_train,y_pred_train)}")

y_pred_test = model.predict(X_test)
print(f"Training score: {r2_score(y_test,y_pred_test)}")
print(f"Absolute error: {mean_absolute_error(y_test,y_pred_test)}")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

d = pd.DataFrame({"x1":[23,34,12,34,23,35],
                  'x2':[5.4,5.6,5,6,4,4.5]})
# d = m1x1 + m2x2

poly_reg = PolynomialFeatures(degree=3)
poly_reg.fit(d)
td = poly_reg.transform(d)

columns = poly_reg.get_feature_names_out()
df  = pd.DataFrame(data=td, columns=columns)
df

poly_reg = PolynomialFeatures(degree=2)
poly_reg.fit(X_train)
X_train_poly = poly_reg.transform(X_train)
X_test_poly = poly_reg.transform(X_test)

X_train_poly.shape, X_test_poly.shape

lr = LinearRegression()

lr.fit(X_train_poly, y_train)

lr.score(X_train_poly, y_train)

lr.score(X_test_poly, y_test)

lr.predict([X_test_poly[0,:]])

y_pred = lr.predict(X_test_poly)
y_pred

y_test

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

mse, rmse

lr.fit(X_train,y_train)

lr.score(X_train,y_train)

lr.score(X_test,y_test)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,SGDRegressor

from sklearn.preprocessing import PolynomialFeatures,StandardScaler

from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)

# y = 0.8x^2 + 0.9x + 2

plt.plot(X, y,'b.')
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# Applying linear regression
lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
r2_score(y_test,y_pred)

plt.plot(X_train,lr.predict(X_train),color='r')
plt.plot(X, y, "b.")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


# Applying Polynomial Linear Regression
# degree 2
poly = PolynomialFeatures(degree=2,include_bias=True)

X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)

print(X_train[0])
print(X_train_trans[0])

lr = LinearRegression()
lr.fit(X_train_trans,y_train)

y_pred_train = lr.predict(X_train_trans)

r2_score(y_train,y_pred_train)

print(lr.coef_)
print(lr.intercept_)

# plt.plot(X_train_trans,y_pred_train,color='r')
plt.plot(X_train, y_train, "b.")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# generating testing data
X_new=np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
y_new = lr.predict(X_new_poly)


plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.plot(X_train, y_train, "b.",label='Training points')
plt.plot(X_test, y_test, "g.",label='Testing points')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

from sklearn.pipeline import Pipeline

def polynomial_regression(degree):
    X_new=np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly.transform(X_new)

    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig,'r', label="Degree " + str(degree), linewidth=2)

    plt.plot(X_train, y_train, "b.", linewidth=3)
    plt.plot(X_test, y_test, "g.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-3, 3, 0, 10])
    plt.show()

polynomial_regression(20)

poly.powers_

# Applying Gradient Descent

poly = PolynomialFeatures(degree=2)

X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)

sgd = SGDRegressor(max_iter=100)
sgd.fit(X_train_trans,y_train)

X_new=np.linspace(-2.9, 2.8, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
y_new = sgd.predict(X_new_poly)

y_pred = sgd.predict(X_test_trans)

plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions " + str(round(r2_score(y_test,y_pred),2)))
plt.plot(X_train, y_train, "b.",label='Training points')
plt.plot(X_test, y_test, "g.",label='Testing points')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 3D polynomial regression
x = 7 * np.random.rand(100, 1) - 2.8
y = 7 * np.random.rand(100, 1) - 2.8

z = x**2 + y**2 + 0.2*x + 0.2*y + 0.1*x*y +2 + np.random.randn(100, 1)
# z = x^2 + y^2 + 0.2x + 0.2y + 0.1xy + 2

import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x=x.ravel(), y=y.ravel(), z=z.ravel())
fig.show()

lr = LinearRegression()
lr.fit(np.array([x,y]).reshape(100,2),z)

x_input = np.linspace(x.min(), x.max(), 10)
y_input = np.linspace(y.min(), y.max(), 10)
xGrid, yGrid = np.meshgrid(x_input,y_input)

final = np.vstack((xGrid.ravel().reshape(1,100),yGrid.ravel().reshape(1,100))).T

z_final = lr.predict(final).reshape(10,10)

import plotly.graph_objects as go

fig = px.scatter_3d(df, x=x.ravel(), y=y.ravel(), z=z.ravel())

fig.add_trace(go.Surface(x = x_input, y = y_input, z =z_final ))

fig.show()

X_multi = np.array([x,y]).reshape(100,2)
X_multi.shape

poly = PolynomialFeatures(degree=30)
X_multi_trans = poly.fit_transform(X_multi)

print("Input",poly.n_features_in_)
print("Ouput",poly.n_output_features_)
print("Powers\n",poly.powers_)

X_multi_trans.shape

lr = LinearRegression()
lr.fit(X_multi_trans,z)

X_test_multi = poly.transform(final)

z_final = lr.predict(X_multi_trans).reshape(10,10)

fig = px.scatter_3d(x=x.ravel(), y=y.ravel(), z=z.ravel())

fig.add_trace(go.Surface(x = x_input, y = y_input, z =z_final))

fig.update_layout(scene = dict(zaxis = dict(range=[0,35])))

fig.show()





