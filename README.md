<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>

# Lowess Model using a Train/test split

Import the necessary libraries:
```Python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from sklearn.metrics import mean_squared_error as mse
from math import ceil
from scipy import linalg
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.model_selection import train_test_split
```
First we define a distance function to calculate the distance between two points; this is important for establishing wich observations go in which neighborhood.
```Python
def dist(u,v):
  if len(v.shape)==1:
    v = v.reshape(1,-1)
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))]) #this subtracts one value from every position in the matrix 
  return d
```

```Python
def dist(u,v):
  if len(v.shape)==1:
    v = v.reshape(1,-1)
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))]) #this subtracts one value from every position in the matrix 
  return d
```

Next we define the actual function that can use train/test splits the code is commented with an explantion of each line:
```Python
def lowess_with_xnew(x, y, xnew,f=2/3, iter=3, intercept=True, qwerty = 6): 
#we a xtrain, ytrain, xtest, a fraction of n to be included in the neighborhoods (f), 
#the number of times to go trhough and remove outliers (iter), and 
  n = len(x) #this gets the length of the x variable, the number of observations
  r = int(ceil(f * n)) # this is the portion of n that will be used in the neigborhoods
  yest = np.zeros(n) #this gives us an array with the same number as the observations filled with zeros to be changed later

  if len(y.shape)==1: # here we reshape y into a matrix if it is not already
    y = y.reshape(-1,1)

  if len(x.shape)==1: #same for x
    x = x.reshape(-1,1)
  
  if intercept:  #if intercept is true, it adds a column of ones to the matrix
    x1 = np.column_stack([np.ones((len(x),1)),x])
  else: #otherwise it does not, this term would act as the 'b' in 'y = mx +b', or the beta_0 
    x1 = x

  h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]
  #we compute the max bounds for the local neighborhoods

  w = np.clip(dist(x,x) / h, 0.0, 1.0) 
  #we take the distance function of x by itself and divide it by the max bounds of the neighborhoods
  #and then remove the values that are at those max bounds or at zero
  #tricubic represents opportunity for mroe research if you want to change kernel
  w = (1 - w ** 3) ** 3
  #this function makes values closer to one become smaller, and values closer to zero bigger, 
  #which makes sense because values closer to the observation should have more weight in the linear regression

  #Looping through all X-points
  delta = np.ones(n) #square matrix filled with ones the same length as n
  for iteration in range(iter): #loops through based on how many times we specified it to cut the outliers
    for i in range(n): #loops trhough every observation and removes outliers
      W = delta * np.diag(w[:,i]) #this is the weights for removing values
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)
      #prediction algorithms
      A = A + 0.0001*np.eye(x1.shape[1]) # if we want L2 regularization (ridge)
      beta = linalg.solve(A, b) #beta is the "solved" matrix for a and b between the independent and dependent variables
      yest[i] = np.dot(x1[i],beta) #set the y estimated values

    residuals = y - yest #calculate residuals 
    s = np.median(np.abs(residuals)) #median of the residuals
    delta = np.clip(residuals / (qwerty * s), -1, 1) #calculate the new array with cut outliers 
    delta = (1 - delta ** 3) ** 3 #assign more importance to observations that gave you less errors and vice versa

  #the following code deals with xtest and ytest
  if x.shape[1]==1:
    f = interp1d(x.flatten(),yest,fill_value='extrapolate')
    output = f(xnew)
  else:
    output = np.zeros(len(xnew))
    for i in range(len(xnew)):
      ind = np.argsort(np.sqrt(np.sum((x-xnew[i])**2,axis=1)))[:r]
      #if you dont extract principle components, you would get an infinite loop
      #use delaunay triangulation 
      pca = PCA(n_components=3)
      x_pca = pca.fit_transform(x[ind])
      tri = Delaunay(x_pca,qhull_options='QJ')
      f = LinearNDInterpolator(tri,y[ind])
      output[i] = f(pca.transform(xnew[i].reshape(1,-1))) # the output may have NaN's where the data points from xnew are outside the convex hull of X
  if sum(np.isnan(output))>0:
    g = NearestNDInterpolator(x,y.ravel()) 
    # output[np.isnan(output)] = g(X[np.isnan(output)])
    output[np.isnan(output)] = g(xnew[np.isnan(output)])
  return output
  ```
  
  Import data to use, split them into train and test sets, and run our function:
  ```Python
  data = pd.read_csv('drive/MyDrive/machineLearning/cars.csv')
  
  x = data.loc[:,'CYL':'WGT'].values
  y = data['MPG'].values
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  
  yhat = lowess_with_xnew(x_train,y_train,x_test,f=1/3,iter=5,intercept=True,qwerty = 2)
  ```
  We can use the mean squared error to determine how well the model runs
  ```Python
  mse(y_test,yhat)
  ```
  Which gives us a value of 17.264771219115353
  
  ## Sklearn Function
  We can also make our function work with sklearn by importing a few functions:
  ```Python
  from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
  ```
  
  Define a class for the function:
  ```Python
  class Lowess_AG_MD:
    def __init__(self, f = 1/10, iter = 3,intercept=True, qwerty = 6):
        self.f = f
        self.iter = iter
        self.intercept = intercept
        self.qwerty = qwerty
    
    def fit(self, x, y):
        f = self.f
        iter = self.iter
        qwerty = self.qwerty
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        f = self.f
        iter = self.iter
        intercept = self.intercept
        qwerty = self.qwerty
        return lowess_with_xnew(x, y, x_new, f, iter, intercept)

    def get_params(self, deep=True):
    # suppose this estimator has parameters "f", "iter" and "intercept"
        return {"f": self.f, "iter": self.iter,"intercept":self.intercept, "qwerty": self.qwerty}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
  ```
  
  And viola!
  ```Python
  model = Lowess_AG_MD(f=1/4,iter=0,intercept=True, qwerty = 4)
  model.fit(x_train,y_train)
  yhat = model.predict(x_test)
  mse(y_test,yhat)
  ```
 Using these parameters, we get a MSE of 22.6786236182753
 
 We can also compare our model to RandomForest with KFold splits
 Import some more modules...
 ```Python
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.model_selection import KFold
 from sklearn.preprocessing import StandardScaler
 ```

Run the model...
  ```Python
  mse_lwr = []
  mse_rf = []
  kf = KFold(n_splits=10,shuffle=True,random_state=1234)
  model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)
  model_lw = Lowess_AG_MD(f=1/2,iter=4,intercept=True,qwerty = 3)
  scale = StandardScaler()


  for idxtrain, idxtest in kf.split(x):
    xtrain = x[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = x[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)

    model_lw.fit(xtrain,ytrain)
    yhat_lw = model_lw.predict(xtest)

    model_rf.fit(xtrain,ytrain)
    yhat_rf = model_rf.predict(xtest)

    mse_lwr.append(mse(ytest,yhat_lw))
    mse_rf.append(mse(ytest,yhat_rf))
  ```
  
  And using these parameters, we get a value of 22.191308168265262 for lowess and 17.11848336742738 for random forest.
