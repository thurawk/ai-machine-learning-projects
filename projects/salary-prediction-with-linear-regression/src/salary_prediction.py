"""
Salary Prediction with Linear Regression
========================================
This project demonstrates supervised learning using Linear Regression to predict
salary based on age, years of experience, and education level.

Author: Thura Win Kyaw
Date: 2025-11-19
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PHASE 1: DATA GENERATION
# ============================================================================

print("=" * 60)
print("PHASE 1: Generating Synthetic Salary Dataset")
print("=" * 60)

def generate_salary_data(n_samples=100):
    """
    Generate synthetic salary data with realistic correlations.
    
    Parameters:
    - n_samples: Number of samples to generate
    
    Returns:
    - DataFrame with Age, Experience, Education, Salary
    """
    # Generate age (22-65 years)
    age = np.random.randint(22, 66, size=n_samples)
    
    # Generate experience (0 to age-22, with some correlation to age)
    experience = np.random.randint(0, np.maximum(1, age - 20), size=n_samples)
    # Add some correlation: older people tend to have more experience
    experience = np.clip(experience + (age - 30) // 3, 0, 40)
    
    # Generate education level (1=High School, 2=Bachelor's, 3=Master's, 4=PhD)
    # Higher age/experience slightly increases chance of higher education
    education_base = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.2, 0.5, 0.25, 0.05])
    # Adjust based on age (older people more likely to have higher education)
    education = education_base.copy()
    for i in range(n_samples):
        if age[i] > 35 and np.random.random() < 0.3:
            education[i] = min(4, education[i] + 1)
    
    # Generate salary with realistic correlations
    # Base salary increases with age, experience, and education
    base_salary = 30000 + (age - 22) * 1500  # Age effect
    base_salary += experience * 2000  # Experience effect
    base_salary += (education - 1) * 15000  # Education effect
    
    # Add some noise for realism (±20%)
    noise = np.random.normal(0, 0.1, size=n_samples)
    salary = base_salary * (1 + noise)
    
    # Ensure salary is within reasonable bounds
    salary = np.clip(salary, 25000, 200000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Experience': experience,
        'Education': education,
        'Salary': salary.astype(int)
    })
    
    # Map education to readable format
    education_map = {1: 'High School', 2: "Bachelor's", 3: "Master's", 4: 'PhD'}
    df['Education_Label'] = df['Education'].map(education_map)
    
    return df

# Generate dataset
n_samples = 100
df = generate_salary_data(n_samples)

# Save dataset
os.makedirs('../data-visualization/data', exist_ok=True)
df.to_csv('../data-visualization/data/salary_data.csv', index=False)
print(f"\n✅ Generated {n_samples} samples")
print(f"✅ Dataset saved to '..data-visualization/data/salary_data.csv'")


# ============================================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 2: Exploratory Data Analysis")
print("=" * 60)

# Correlation analysis
print("\n📊 Correlation Matrix:")
correlation_matrix = df[['Age', 'Experience', 'Education', 'Salary']].corr()
print(correlation_matrix)

# Create visualizations directory
os.makedirs('../data-visualization/results', exist_ok=True)

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap: Features vs Salary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../data-visualization/results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: correlation_heatmap.png")
plt.close()

# 2. Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

axes[0, 0].hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(df['Experience'], bins=20, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Experience Distribution')
axes[0, 1].set_xlabel('Years of Experience')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(df['Education'], bins=4, color='coral', edgecolor='black')
axes[1, 0].set_title('Education Level Distribution')
axes[1, 0].set_xlabel('Education Level')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_xticks([1, 2, 3, 4])
axes[1, 0].set_xticklabels(['HS', "Bach", "Mast", 'PhD'])

axes[1, 1].hist(df['Salary'], bins=20, color='gold', edgecolor='black')
axes[1, 1].set_title('Salary Distribution')
axes[1, 1].set_xlabel('Salary ($)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('../data-visualization/results/feature_distributions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: feature_distributions.png")
plt.close()

# 3. Pair plot
plt.figure(figsize=(12, 10))
sns.pairplot(df[['Age', 'Experience', 'Education', 'Salary']], 
             diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot: Feature Relationships', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('../data-visualization/results/pair_plot.png', dpi=300, bbox_inches='tight')
print("✅ Saved: pair_plot.png")
plt.close()

# ============================================================================
# PHASE 3: MODEL DEVELOPMENT
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 3: Model Development")
print("=" * 60)

# Prepare features and target
X = df[['Age', 'Experience', 'Education']]
y = df['Salary']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Feature scaling (optional for linear regression, but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Simple Linear Regression (Age only)
print("\n" + "-" * 60)
print("Model 1: Simple Linear Regression (Age → Salary)")
print("-" * 60)

model_simple = LinearRegression()
X_train_age = X_train[['Age']].values
X_test_age = X_test[['Age']].values
model_simple.fit(X_train_age, y_train)
y_pred_simple = model_simple.predict(X_test_age)

r2_simple = r2_score(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
mae_simple = mean_absolute_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)

print(f"R² Score: {r2_simple:.4f}")
print(f"MSE: {mse_simple:.2f}")
print(f"MAE: ${mae_simple:.2f}")
print(f"RMSE: ${rmse_simple:.2f}")

# Model 2: Multiple Linear Regression (All features)
print("\n" + "-" * 60)
print("Model 2: Multiple Linear Regression (Age + Experience + Education → Salary)")
print("-" * 60)

model_multiple = LinearRegression()
model_multiple.fit(X_train_scaled, y_train)
y_pred_multiple = model_multiple.predict(X_test_scaled)

r2_multiple = r2_score(y_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
mae_multiple = mean_absolute_error(y_test, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)

print(f"R² Score: {r2_multiple:.4f}")
print(f"MSE: {mse_multiple:.2f}")
print(f"MAE: ${mae_multiple:.2f}")
print(f"RMSE: ${rmse_multiple:.2f}")

# Feature coefficients
print("\n📈 Feature Coefficients (Multiple Regression):")
feature_names = ['Age', 'Experience', 'Education']
for name, coef in zip(feature_names, model_multiple.coef_):
    print(f"  {name}: {coef:.2f}")
print(f"  Intercept: {model_multiple.intercept_:.2f}")

# Cross-validation
print("\n" + "-" * 60)
print("Cross-Validation (5-fold)")
print("-" * 60)
cv_scores = cross_val_score(model_multiple, X_train_scaled, y_train, 
                           cv=5, scoring='r2')
print(f"CV R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# PHASE 4: ADVANCED FEATURES - POLYNOMIAL REGRESSION
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 4: Polynomial Regression")
print("=" * 60)

# Polynomial features (degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)

r2_poly = r2_score(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

print(f"R² Score: {r2_poly:.4f}")
print(f"MSE: {mse_poly:.2f}")
print(f"MAE: ${mae_poly:.2f}")
print(f"RMSE: ${rmse_poly:.2f}")

# Model Comparison
print("\n" + "-" * 60)
print("Model Comparison")
print("-" * 60)
comparison_df = pd.DataFrame({
    'Model': ['Simple LR (Age only)', 'Multiple LR', 'Polynomial LR (deg=2)'],
    'R² Score': [r2_simple, r2_multiple, r2_poly],
    'RMSE': [rmse_simple, rmse_multiple, rmse_poly],
    'MAE': [mae_simple, mae_multiple, mae_poly]
})
print(comparison_df.to_string(index=False))

# Select best model
best_model = model_multiple if r2_multiple >= r2_poly else model_poly
best_name = "Multiple Linear Regression" if r2_multiple >= r2_poly else "Polynomial Regression"
print(f"\n🏆 Best Model: {best_name} (R² = {max(r2_multiple, r2_poly):.4f})")

# ============================================================================
# PHASE 5: VISUALIZATION
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 5: Visualization")
print("=" * 60)

# 1. Scatter plot with regression line (Age vs Salary)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test['Age'], y_test, alpha=0.6, color='blue', label='Actual')
plt.scatter(X_test['Age'], y_pred_simple, alpha=0.6, color='red', label='Predicted (Simple)')
# Regression line
age_range = np.linspace(X_test['Age'].min(), X_test['Age'].max(), 100).reshape(-1, 1)
salary_range = model_simple.predict(age_range)
plt.plot(age_range, salary_range, 'g--', linewidth=2, label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Salary ($)')
plt.title('Simple Linear Regression: Age vs Salary')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Prediction vs Actual (Multiple Regression)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_multiple, alpha=0.6, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Salary ($)')
plt.ylabel('Predicted Salary ($)')
plt.title(f'Multiple Linear Regression\nR² = {r2_multiple:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data-visualization/results/regression_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: regression_analysis.png")
plt.close()

# 3. Residual plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

residuals_multiple = y_test - y_pred_multiple
residuals_poly = y_test - y_pred_poly

axes[0].scatter(y_pred_multiple, residuals_multiple, alpha=0.6, color='blue')
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Salary ($)')
axes[0].set_ylabel('Residuals ($)')
axes[0].set_title('Residual Plot: Multiple Linear Regression')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_pred_poly, residuals_poly, alpha=0.6, color='green')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Salary ($)')
axes[1].set_ylabel('Residuals ($)')
axes[1].set_title('Residual Plot: Polynomial Regression')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data-visualization/results/residual_plots.png', dpi=300, bbox_inches='tight')
print("✅ Saved: residual_plots.png")
plt.close()

# 4. Model Comparison Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# R² Comparison
axes[0].bar(['Simple LR', 'Multiple LR', 'Polynomial LR'], 
            [r2_simple, r2_multiple, r2_poly], 
            color=['skyblue', 'lightgreen', 'coral'])
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score Comparison')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([r2_simple, r2_multiple, r2_poly]):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# RMSE Comparison
axes[1].bar(['Simple LR', 'Multiple LR', 'Polynomial LR'], 
            [rmse_simple, rmse_multiple, rmse_poly], 
            color=['skyblue', 'lightgreen', 'coral'])
axes[1].set_ylabel('RMSE ($)')
axes[1].set_title('RMSE Comparison (Lower is Better)')
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([rmse_simple, rmse_multiple, rmse_poly]):
    axes[1].text(i, v + 500, f'${v:.0f}', ha='center', fontweight='bold')

# MAE Comparison
axes[2].bar(['Simple LR', 'Multiple LR', 'Polynomial LR'], 
            [mae_simple, mae_multiple, mae_poly], 
            color=['skyblue', 'lightgreen', 'coral'])
axes[2].set_ylabel('MAE ($)')
axes[2].set_title('MAE Comparison (Lower is Better)')
axes[2].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([mae_simple, mae_multiple, mae_poly]):
    axes[2].text(i, v + 500, f'${v:.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../data-visualization/results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison.png")
plt.close()

# ============================================================================
# PHASE 6: MODEL PERSISTENCE & PREDICTIONS
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 6: Model Persistence & Predictions")
print("=" * 60)

# Save models
os.makedirs('../data-visualization/models', exist_ok=True)

# Save best model (Multiple Linear Regression)
joblib.dump(model_multiple, '../data-visualization/models/linear_regression_model.pkl')
joblib.dump(scaler, '../data-visualization/models/scaler.pkl')
print("✅ Saved: linear_regression_model.pkl")
print("✅ Saved: scaler.pkl")

# Save polynomial model
joblib.dump(model_poly, '../data-visualization/models/polynomial_regression_model.pkl')
joblib.dump(poly_features, '../data-visualization/models/poly_features.pkl')
print("✅ Saved: polynomial_regression_model.pkl")
print("✅ Saved: poly_features.pkl")

# Load and test saved model
print("\n" + "-" * 60)
print("Testing Saved Model")
print("-" * 60)
loaded_model = joblib.load('../data-visualization/models/linear_regression_model.pkl')
loaded_scaler = joblib.load('../data-visualization/models/scaler.pkl')

# Example predictions
print("\n📊 Example Predictions:")
print("-" * 60)
example_data = pd.DataFrame({
    'Age': [25, 35, 45, 30],
    'Experience': [2, 10, 20, 5],
    'Education': [2, 3, 4, 2]  # 2=Bachelor's, 3=Master's, 4=PhD
})

example_scaled = loaded_scaler.transform(example_data)
predictions = loaded_model.predict(example_scaled)

for i, (age, exp, edu, pred) in enumerate(zip(
    example_data['Age'], 
    example_data['Experience'], 
    example_data['Education'], 
    predictions
)):
    edu_label = {1: 'High School', 2: "Bachelor's", 3: "Master's", 4: 'PhD'}[edu]
    print(f"\nExample {i+1}:")
    print(f"  Age: {age}, Experience: {exp} years, Education: {edu_label}")
    print(f"  Predicted Salary: ${pred:,.2f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n✅ Dataset: {n_samples} samples generated and saved")
print(f"✅ Models Trained: Simple LR, Multiple LR, Polynomial LR")
print(f"✅ Best Model: {best_name} (R² = {max(r2_multiple, r2_poly):.4f})")
print(f"✅ Visualizations: 6 plots saved to results/")
print(f"✅ Models Saved: Ready for deployment")
print("\n" + "=" * 60)
print("Project Enhancement Complete! 🎉")
print("=" * 60)
