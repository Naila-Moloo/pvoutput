import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn. model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Imputing missing values and scaling values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# accuracy
from sklearn.metrics import accuracy_score

# LIME for explaining predictions
import lime
import lime.lime_tabular

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Visualizing a Decision Tree
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import plot_tree

# Read in data into a dataframe
data = pd.read_csv('/Users/nailamolooicloud.com/Downloads/spg.csv')
data.head

# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
 "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

missing_values_table(data)

# Rename the output
data = data.rename(columns = {'generated_power_kw': 'power output'})

# Histogram of the Generated Power Output
plt.hist(data['power output'].dropna(), bins = 100, edgecolor='black');
plt.xlabel('Power Output'); plt.ylabel('Count');
plt.title('Generated Power Output Distribution');

# Find all correlations and sort
correlations_data = data.corr()['power output'].sort_values()

# Print the most negative correlations
print(correlations_data.head(15), '\n')

# Print the most positive correlations
print(correlations_data.tail(15))

# Pairs Plot
features = pd.concat([data], axis = 1)

# Extract the columns to  plot
plot_data = features[['power output', 'zenith',
                      'angle_of_incidence',
                      'temperature_2_m_above_gnd']]

# Replace the inf with nan
plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})

# Rename columns
plot_data = plot_data.rename(columns = {'angle_of_incidence': 'angle of incidence',
                                        'temperature_2_m_above_gnd': 'temperature above ground',
                                        })
# Drop na values
plot_data = plot_data.dropna()

# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

# Create the pairgrid object
grid = sns.PairGrid(data = plot_data, size = 3)

# Upper is a scatter plot
grid.map_upper(plt.scatter, color = 'blue', alpha = 0.6)

# Diagonal is a histogram
grid.map_diag(plt.hist, color = 'blue', edgecolor = 'white')

# Bottom is correlation and density plot
grid.map_lower(corr_func);
grid.map_lower(sns.kdeplot, cmap = plt.cm.Blues)


# Copy the original data
features = data.copy()

# Select the numeric columns
numeric_subset = data.select_dtypes('number')

# Create columns with log of numeric columns
for col in numeric_subset.columns:
    # Skip the power output column
    if col == 'power output':
        next
    else:
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

features.shape

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        threshold: any features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Dont want to remove correlations between Energy Star Score
    y = x['power output']
    x = x.drop(columns=['power output'])

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    x = x.drop(columns=['total_cloud_cover_sfc',
                    'wind_speed_10_m_above_gnd' ])

    # Add the score back in to the data
    x['power output'] = y
    return x
# Remove the collinear features above a specified correlation coefficient
features = remove_collinear_features(features, 0.6);
# Remove any columns with all na values
features = features.dropna(axis=1, how = 'all')
print(features.shape)

# Separate out the features and targets
features = data.drop(columns='power output')
targets = pd.DataFrame(data['power output'])

# Replace the inf and -inf with nan
features = features.replace({np.inf: np.nan, -np.inf: np.nan})

# Split into 70% training and 30% testing set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
baseline_guess = np.median(y_train)

print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))

# Create an imputer object with a median filling strategy
imputer = SimpleImputer(strategy='median')

# Train on the training features
imputer.fit(X_train)

# Transform both training data and testing data
X = imputer.transform(X_train)
X_test = imputer.transform(X_test)

'Missing values in training features: ', np.sum(np.isnan(X_train))
'Missing values in testing features:  ', np.sum(np.isnan(X_test))

# Make sure all values are finite
np.where(~np.isfinite(X_train))
np.where(~np.isfinite(X_test))

# Create the scaler object with a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X_train)

# Transform the training and testing data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Convert y to one-dimensional array (vector)
y_train = np.array(y_train).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)

    # Return the performance metric
    return model_mae

    models = []

lr = LinearRegression()
lr_mae = fit_and_evaluate(lr)

print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)

svm = SVR(C = 1.0, gamma = 0.1)
svm_mae = fit_and_evaluate(svm)

print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)

random_forest = RandomForestRegressor(random_state=42)
random_forest_mae = fit_and_evaluate(random_forest)

print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)

gradient_boosted = GradientBoostingRegressor(random_state=42)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)

print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)

knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)

print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)

# print a fivethirtyeight graph
plt.style.use('fivethirtyeight')

# Dataframe to hold the results
model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Support Vector Machine',
                                           'Random Forest', 'Gradient Boosted',
                                            'K-Nearest Neighbors'],
                                 'mae': [lr_mae, svm_mae, random_forest_mae,
                                         gradient_boosted_mae, knn_mae]})

# Horizontal bar chart of test mae
model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh')

# Plot formatting
plt.xlabel('Mean Absolute Error')
plt.ylabel('')
plt.yticks(fontsize=6)

# hyperparameters
# Number of trees used in the boosting process
n_estimators = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

# Maximum depth of each tree
max_depth = [2, 4, 5, 8, 12]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 8, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

# Create the model to use for hyperparameter tuning
model = RandomForestRegressor(random_state = 42)

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25,
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1,
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(X_train, y_train)

# Get all of the cv results and sort by the test performance
random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  random_results.head(10)


# Default model
default_model = RandomForestRegressor(random_state = 42)

# Final model
final_model = RandomForestRegressor(
                              max_depth = 14,
                              min_samples_leaf = 1,
                              min_samples_split = 8,
                              max_features = 'log2',
                              random_state = 42, n_estimators = 900 )

import timeit
default_model.fit(X_train, y_train)
print("The time taken is ",timeit.timeit(stmt='a=10;b=10;sum=a+b'))
final_model.fit(X_train, y_train)
print("The time taken is ",timeit.timeit(stmt='a=10;b=10;sum=a+b'))

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set using default and final model
default_pred = default_model.predict(X_test)
final_pred = final_model.predict(X_test)

print('Default model performance on the test set: MAE = %0.4f.' % mae(y_test, default_pred))
print('Final model performance on the test set:   MAE = %0.4f.' % mae(y_test, final_pred))

# Evaluate using a train and a test set
model = RandomForestRegressor()
model.fit(X_train, y_train)
result = final_model.score(X_test, y_test)
print("Final Accuracy: %.2f%%" % (result*100.0))

# Evaluate using a train and a test set
model = RandomForestRegressor()
model.fit(X_train, y_train)
result = default_model.score(X_test, y_test)
print("Default Accuracy: %.2f%%" % (result*100.0))

importances = model.feature_importances_
indices = np.argsort(importances)

# Keep only the most important features
X_reduced = X[:, indices]
X_test_reduced = X_test[:, indices]

print('Most important training features shape: ', X_reduced.shape)
print('Most important testing  features shape: ', X_test_reduced.shape)

rf = RandomForestRegressor()

# Fit on full set of features
rf.fit(X_train, y_train)
rf_full_pred = rf.predict(X_test)

# Fit on reduced set of features
rf.fit(X_reduced, y_train)
rf_reduced_pred = rf.predict(X_test_reduced)

# Display results
print('Random Forest Regression Full Results: MAE =    %0.4f.' % mae(y_test, rf_full_pred))
print('Random Forest Regression Reduced Results: MAE = %0.4f.' % mae(y_test, rf_reduced_pred))

# Find the residuals
residuals = abs(final_pred - y_test)

# Exact the worst and best prediction
wrong = X_test[np.argmax(residuals), :]
right = X_test[np.argmin(residuals), :]

# Create a lime explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(training_data = X_train,
                                                   mode = 'regression',
                                                   training_labels = y_train,
                                                   feature_names = list(features))

# Display the predicted and true value for the wrong instance
print('Prediction: %0.4f' % model.predict(wrong.reshape(1, -1)))
print('Actual Value: %0.4f' % y_test[np.argmax(residuals)])

# Explanation for wrong prediction
wrong_exp = explainer.explain_instance(data_row = wrong,
                                       predict_fn = model.predict)

# Plot the prediction explanation
wrong_exp.as_pyplot_figure();
plt.yticks(fontsize=5)
plt.title('Explanation of Wrong Prediction');
plt.xlabel('Effect on Prediction');

# Display the predicted and true value for the right instance
print('Prediction: %0.4f' % model.predict(right.reshape(1, -1)))
print('Actual Value: %0.4f' % y_test[np.argmin(residuals)])

# Explanation for right prediction
right_exp = explainer.explain_instance(right, model.predict)
right_exp.as_pyplot_figure();
plt.yticks(fontsize=5)
plt.title('Explanation of Right Prediction');
plt.xlabel('Effect on Prediction');

X = data
y = targets

regr = DecisionTreeRegressor(max_depth=3, random_state=42)
model = regr.fit(X, y)

text_representation = tree.export_text(regr)
print(text_representation)

fig = plt.figure(figsize=(12,20))
_ = tree.plot_tree(regr, filled=True)
plt.show()
