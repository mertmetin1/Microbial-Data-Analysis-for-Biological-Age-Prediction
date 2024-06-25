from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVR
import seaborn as sns


# Load the data
ages = pd.read_csv('data/Ages.csv')
microbial_data = pd.read_csv('data/data.csv')

# Merge datasets on sample names
data = pd.merge(ages, microbial_data, on='Sample Accession')

# Drop rows with null values
data.dropna(inplace=True)

# Drop the 'Sample Accession.1' column
data.drop('Sample Accession.1', axis=1, inplace=True)

# Features and target variable
X = data.iloc[:, 2:]  # Independent variables
y = data.iloc[:, 1]   # Target variable (biological age)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectFromModel(RandomForestRegressor(random_state=42), max_features=100)
X_selected = selector.fit_transform(X_scaled, y)

# Create DataFrame with selected features
selected_feature_names = data.columns[2:][selector.get_support()].tolist()
X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)

# Calculate correlation matrix
correlation_matrix = X_selected_df.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Selected Features')
plt.show()

# PCA ile veri boyutunu azaltma
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_selected_df)





# Verileri eğitim ve test setlerine bölmek
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)









# Hyperparameter Tuning for SVM
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

svm_model = SVR()
svm_grid_search = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
svm_grid_search.fit(X_train_pca, y_train_pca)
best_svm_model = svm_grid_search.best_estimator_
best_svm_y_pred = best_svm_model.predict(X_test_pca)
best_svm_mae = mean_absolute_error(y_test_pca, best_svm_y_pred)
best_svm_r_squared = r2_score(y_test_pca, best_svm_y_pred)

# Hyperparameter Tuning for Gradient Boosted Trees
gbt_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}

gbt_model = GradientBoostingRegressor(random_state=42)
gbt_grid_search = GridSearchCV(estimator=gbt_model, param_grid=gbt_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
gbt_grid_search.fit(X_train_pca, y_train_pca)
best_gbt_model = gbt_grid_search.best_estimator_
best_gbt_y_pred = best_gbt_model.predict(X_test_pca)
best_gbt_mae = mean_absolute_error(y_test_pca, best_gbt_y_pred)
best_gbt_r_squared = r2_score(y_test_pca, best_gbt_y_pred)

# Hyperparameter Tuning for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train_pca, y_train_pca)
best_rf_model = rf_grid_search.best_estimator_
best_rf_y_pred = best_rf_model.predict(X_test_pca)
best_rf_mae = mean_absolute_error(y_test_pca, best_rf_y_pred)
best_rf_r_squared = r2_score(y_test_pca, best_rf_y_pred)

# Ensemble Model
ensemble_model = VotingRegressor(estimators=[
    ('rf', best_rf_model),
    ('gbr', best_gbt_model),
    ('svm', best_svm_model)

])

ensemble_model.fit(X_train_pca, y_train_pca)
ensemble_y_pred = ensemble_model.predict(X_test_pca)
ensemble_mae = mean_absolute_error(y_test_pca, ensemble_y_pred)
ensemble_r_squared = r2_score(y_test_pca, ensemble_y_pred)


# Cross-validated results
best_svm_mae_cv = -cross_val_score(best_svm_model, X_selected, y, cv=5, scoring='neg_mean_absolute_error').mean()
best_svm_r_squared_cv = cross_val_score(best_svm_model, X_selected, y, cv=5, scoring='r2').mean()

best_gbt_mae_cv = -cross_val_score(best_gbt_model, X_selected, y, cv=5, scoring='neg_mean_absolute_error').mean()
best_gbt_r_squared_cv = cross_val_score(best_gbt_model, X_selected, y, cv=5, scoring='r2').mean()

best_rf_mae_cv = -cross_val_score(best_rf_model, X_selected, y, cv=5, scoring='neg_mean_absolute_error').mean()
best_rf_r_squared_cv = cross_val_score(best_rf_model, X_selected, y, cv=5, scoring='r2').mean()

ensemble_mae_cv = -cross_val_score(ensemble_model, X_pca, y, cv=5, scoring='neg_mean_absolute_error').mean()
ensemble_r_squared_cv = cross_val_score(ensemble_model, X_pca, y, cv=5, scoring='r2').mean()



import matplotlib.pyplot as plt

# Plotting real vs predicted ages for each model
plt.figure(figsize=(20, 16))



# SVM Model
plt.subplot(2, 3, 3)  # Corrected subplot index
plt.scatter(y_test, best_svm_y_pred, alpha=0.5, label='SVM Predictions', color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='black')
plt.xlabel('Real Age')
plt.ylabel('Predicted Age')
plt.title('SVM: Real Age vs Predicted Age')
plt.legend()

# GBT Model
plt.subplot(2, 3, 4)  # Corrected subplot index
plt.scatter(y_test, best_gbt_y_pred, alpha=0.5, label='GBT Predictions', color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='black')
plt.xlabel('Real Age')
plt.ylabel('Predicted Age')
plt.title('Gradient Boosted Trees: Real Age vs Predicted Age')
plt.legend()

# Random Forest Model
plt.subplot(2, 3, 5)  # Corrected subplot index
plt.scatter(y_test, best_rf_y_pred, alpha=0.5, label='Random Forest Predictions', color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='black')
plt.xlabel('Real Age')
plt.ylabel('Predicted Age')
plt.title('Random Forest: Real Age vs Predicted Age')
plt.legend()

# Ensemble Model
plt.subplot(2, 3, 6)  # Corrected subplot index
plt.scatter(y_test_pca, ensemble_y_pred, alpha=0.5, label='Ensemble Predictions', color='b')
plt.plot([y_test_pca.min(), y_test_pca.max()], [y_test_pca.min(), y_test_pca.max()], '--', color='black')
plt.xlabel('Real Age')
plt.ylabel('Predicted Age')
plt.title('Ensemble Model: Real Age vs Predicted Age')
plt.legend()

plt.tight_layout()
plt.show()




# Model Performansı (Test Verileri)
print("Model Performance on Test Data:")
print(f"SVM: MAE = {best_svm_mae:.2f}  R-squared = {best_svm_r_squared:.2f}  Accuracy = {100 * (1 - best_svm_mae / y_test.mean()):.2f}%")
print(f"Gradient Boosted Trees Regressor: MAE = {best_gbt_mae:.2f}  R-squared = {best_gbt_r_squared:.2f}  Accuracy = {100 * (1 - best_gbt_mae / y_test.mean()):.2f}%")
print(f"Random Forest Regressor: MAE = {best_rf_mae:.2f}  R-squared = {best_rf_r_squared:.2f}  Accuracy = {100 * (1 - best_rf_mae / y_test.mean()):.2f}%")
print(f"Ensemble Model (RF, GBT, SVM): MAE = {ensemble_mae:.2f}  R-squared = {ensemble_r_squared:.2f}  Accuracy = {100 * (1 - ensemble_mae / y_test_pca.mean()):.2f}%")

# Çapraz Doğrulama Sonuçları
print("\nCross-validated results:")
print(f"SVM: Cross-validated MAE = {best_svm_mae_cv:.2f}  Cross-validated R-squared = {best_svm_r_squared_cv:.2f}  Cross-validated Accuracy = {100 * (1 - best_svm_mae_cv / y.mean()):.2f}%")
print(f"Gradient Boosted Trees Regressor: Cross-validated MAE = {best_gbt_mae_cv:.2f}  Cross-validated R-squared = {best_gbt_r_squared_cv:.2f}  Cross-validated Accuracy = {100 * (1 - best_gbt_mae_cv / y.mean()):.2f}%")
print(f"Random Forest Regressor: Cross-validated MAE = {best_rf_mae_cv:.2f}  Cross-validated R-squared = {best_rf_r_squared_cv:.2f}  Cross-validated Accuracy = {100 * (1 - best_rf_mae_cv / y.mean()):.2f}%")
print(f"Ensemble Model (RF, GBT, SVM): Cross-validated MAE = {ensemble_mae_cv:.2f}  Cross-validated R-squared = {ensemble_r_squared_cv:.2f}  Cross-validated Accuracy = {100 * (1 - ensemble_mae_cv / y.mean()):.2f}%")
