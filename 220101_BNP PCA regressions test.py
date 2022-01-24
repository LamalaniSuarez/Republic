#!/usr/bin/env python
# coding: utf-8

# In[1]:


path_name = '/Users/lamal/S4 Dropbox/Lamalani Suarez/for_shared_notebooks/211020_BNP/'


# In[3]:


import sys
sys.path.append(path_name)

import pandas as pd
import numpy as np

# h5py is used to access the raw data recorded by the Subterra Green.
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

# This is my (s4) module
import s4.data as d

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from sklearn import preprocessing

from sklearn.pipeline import Pipeline


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from sklearn import metrics

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.svm import NuSVR
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


# In[4]:


def rms_error(actual, predicted):
    mse = metrics.mean_squared_error(actual, predicted)
    return np.sqrt(mse)

def neg_rmse_score(actual, predicted):
    return -rms_error(actual, predicted)

def neg_rmse_scorer(model, features, y_true):
    y_predicted = model.predict(features)
    return neg_rmse_score(y_true, y_predicted)

def compute_rpiq(actual, predicted):
    q3 = np.percentile(actual, 75)
    q1 = np.percentile(actual, 25)
    rmse = rms_error(actual, predicted)
    rpiq = (q3 - q1) / rmse
    return rpiq


# In[5]:


df = pd.read_csv(path_name + 'spectra_and_chem_merged_10cm_211207.csv')

df.dropna(subset=['force'], inplace=True)


# In[6]:


df


# In[7]:


vis_waves = d.read_wavelength_vector(path_name + 'BNP_211020.h5', 'session001/cal001', 1)
vis_columns = [f'V{wave:0.1f}' for wave in vis_waves]
ftir_waves = d.read_wavelength_vector(path_name + 'BNP_211020.h5', 'session001/cal001', 2)
ftir_columns = [f'F{wave:0.1f}' for wave in ftir_waves]
print(len(vis_columns), len(ftir_columns))


# In[8]:


vis_keep_waves = list(range(500, 1000, 10))
ftir_keep_waves = list(range(1200, 2400, 10))
print(len(vis_keep_waves), len(ftir_keep_waves))


vis_keep_columns = []
ftir_keep_columns = []
for wave in vis_keep_waves:    
    position = d.find_position_in_wavelength_vector(vis_waves, wave)
    column = vis_columns[position]
    vis_keep_columns.append(column)
for wave in ftir_keep_waves:
    position = d.find_position_in_wavelength_vector(ftir_waves, wave)
    column = ftir_columns[position]
    ftir_keep_columns.append(column)
print(len(vis_keep_columns), len(ftir_keep_columns))


# In[9]:


X = df[vis_keep_columns + ftir_keep_columns].copy()
y = df['OC'].copy()


# In[10]:


X_train_full, X_test, y_train_full, y_test = train_test_split(X, y,
                                                   test_size=0.20,
                                                   random_state=77)

X_train, X_val, y_train, y_val = train_test_split(X_train_full,
                                                  y_train_full,
                                                  test_size=0.20,
                                                  random_state=77)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


# In[13]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
    index = n_components - 1

    scaler = ('standard scaler', preprocessing.StandardScaler())
    pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
    reg = ('knn_2', KNeighborsRegressor(n_neighbors=2))
    kfold = KFold(n_splits=5, shuffle=True, random_state=77)

    r2s = []
    rmses = []
    y_preds = []

    steps = [scaler,
            pca,
            reg]

    pipe = Pipeline(steps=steps)
    
    cv_results = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=kfold,
        scoring=scoring,
        error_score='raise'
    )

    r2s.append(cv_results['test_r2'])
    rmses.append(cv_results['test_neg_rmse'])

    print()
    print('***************')
    print(f'{n_components} COMPONENTS')
    fits_df.loc[index, 'n_components'] = n_components
    print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
    fits_df.loc[index, 'r2cv'] = np.mean(r2s)
    print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
    fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    y_preds.append(y_pred)

    val_rmse = rms_error(y_val, y_pred)
    val_r2 = metrics.r2_score(y_val, y_pred)

    fits_df.loc[index, 'r2val'] = val_r2
    fits_df.loc[index, 'rmseval'] = val_rmse

    print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
    print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

    regressor_name = 'PCA/linear regression'
    max_val = df['OC'].max()
    
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.plot(y_val, y_preds[0], 'o')
    ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
    ax.set_xlabel('Actual % OC')
    ax.set_ylabel('Predicted % OC')
    ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
    ax.set_ylim(0, max_val + 1.0)
    ax.set_xlim(0, max_val + 1.0)
    ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
    ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
    ax.legend()
    plt.show()
    plt.close()
    
fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[17]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('knn_3', KNeighborsRegressor(n_neighbors=3))
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()
    
fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[21]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('knn_4', KNeighborsRegressor(n_neighbors=4))
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[22]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('knn_5', KNeighborsRegressor(n_neighbors=5))
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[23]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('knn_6', KNeighborsRegressor(n_neighbors=6))
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[27]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('linear', LinearRegression())
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[30]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('SVR', SVR())
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[31]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('NuSVR', NuSVR())
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[32]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('DecisionTree1', DecisionTreeRegressor(max_depth=1))
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[33]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('DecisionTree2', DecisionTreeRegressor(max_depth=2))
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[36]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('DecisionTree3', DecisionTreeRegressor(max_depth=3))
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[39]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  pca = ('pca', PCA(n_components=n_components, svd_solver='full'))
  reg = ('DecisionTree4', DecisionTreeRegressor(max_depth=4))
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,
          pca,
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[40]:


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'neg_rmse': neg_rmse_scorer
}

fits_df = pd.DataFrame(columns=['n_components', 'r2cv', 'rmsecv', 'r2val', 'rmseval'])


for n_components in range(1, 40):
  index = n_components - 1
  
  scaler = ('standard scaler', preprocessing.StandardScaler())
  
  reg = ('pcr', PLSRegression(n_components=n_components))
  kfold = KFold(n_splits=5, shuffle=True, random_state=77)

  r2s = []
  rmses = []
  y_preds = []

  steps = [scaler,          
          reg]

  pipe = Pipeline(steps=steps)

  cv_results = cross_validate(
      pipe,
      X_train,
      y_train,
      cv=kfold,
      scoring=scoring,
      error_score='raise'
  )
    
  r2s.append(cv_results['test_r2'])
  rmses.append(cv_results['test_neg_rmse'])

  print()
  print('***************')
  print(f'{n_components} COMPONENTS')
  fits_df.loc[index, 'n_components'] = n_components
  print(f'Mean r2 of cross-validation for {n_components} components: {np.mean(r2s):0.2f}')
  fits_df.loc[index, 'r2cv'] = np.mean(r2s)
  print(f'Mean RMSE of cross-validation for {n_components} components: {-np.mean(rmses):0.2f}')
  fits_df.loc[index, 'rmsecv'] = -np.mean(rmses)

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_val)
  y_preds.append(y_pred)

  val_rmse = rms_error(y_val, y_pred)
  val_r2 = metrics.r2_score(y_val, y_pred)

  fits_df.loc[index, 'r2val'] = val_r2
  fits_df.loc[index, 'rmseval'] = val_rmse

  print(f'R2 for validation set for {n_components} components:  {val_r2:0.2f}')
  print(f'RMSE for validation set for {n_components} components: {val_rmse:0.2f}')

  regressor_name = 'PCA/linear regression'
  max_val = df['OC'].max()

  fig, ax = plt.subplots(figsize = (8, 8))
  ax.plot(y_val, y_preds[0], 'o')
  ax.plot([0, 16], [0, 16], color='gray', alpha=0.2, label='1:1 line')
  ax.set_xlabel('Actual % OC')
  ax.set_ylabel('Predicted % OC')
  ax.set_title(f'Actual vs. Predicted for validation set, {reg[0]} regression, {n_components} components')
  ax.set_ylim(0, max_val + 1.0)
  ax.set_xlim(0, max_val + 1.0)
  ax.text(0.5, max_val + 0.25, f'RMSE: {val_rmse:0.2f}')
  ax.text(0.5, max_val - 0.25 , f'R2: {val_r2:0.2f}')
  ax.legend()
  plt.show()
  plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['r2cv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['r2val'], label='validation set')
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('R2 of cross validation')
ax.set_title(f'R2, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(fits_df['n_components'], fits_df['rmsecv'], label='cross validation')
ax.plot(fits_df['n_components'], fits_df['rmseval'], label='validation set')
ax.set_ylim(0.0, 2.0)
ax.set_xlabel('Number of components')
ax.set_ylabel('RMSE of cross validation')
ax.set_title(f'RMSE, with regressor: {reg[0]}')
plt.legend()
plt.show()
plt.close()


# In[ ]:




