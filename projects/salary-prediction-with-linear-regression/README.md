# 💰 Salary Prediction with Linear Regression

## 📌 Overview

This project demonstrates **supervised machine learning** using Linear Regression to predict salary based on demographic and professional features. The project covers the complete ML pipeline from data generation to model deployment, including multiple regression techniques and comprehensive model evaluation.

## 🎯 Objectives

- Build Linear Regression models to predict salary
- Implement Simple and Multiple Linear Regression
- Explore Polynomial Regression for non-linear relationships
- Perform comprehensive model evaluation and comparison
- Visualize regression results and model performance
- Save and deploy trained models for predictions

## 📊 Dataset

The project uses a synthetic salary dataset with realistic correlations:

- **Age**: 22-65 years
- **Experience**: Years of professional experience (0-40 years)
- **Education**: Education level (1=High School, 2=Bachelor's, 3=Master's, 4=PhD)
- **Salary**: Annual salary in dollars (target variable)

**Dataset Size**: 100 samples (80% training, 20% testing)

## 🚀 Features

### Machine Learning Models

- ✅ **Simple Linear Regression**: Predict salary from age only
- ✅ **Multiple Linear Regression**: Predict salary from age, experience, and education
- ✅ **Polynomial Regression**: Capture non-linear relationships (degree 2)
- ✅ **Model Comparison**: Evaluate and compare all models

### Data Analysis

- ✅ **Exploratory Data Analysis (EDA)**:
  - Statistical summaries
  - Correlation analysis
  - Feature distributions
  - Pair plots
- ✅ **Data Preprocessing**:
  - Feature scaling
  - Train-test split
  - Feature engineering

### Model Evaluation

- ✅ **Metrics**:
  - R² Score (coefficient of determination)
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- ✅ **Cross-Validation**: 5-fold cross-validation
- ✅ **Residual Analysis**: Check model assumptions

### Visualizations

- ✅ Correlation heatmap
- ✅ Feature distribution plots
- ✅ Pair plots for feature relationships
- ✅ Regression lines and scatter plots
- ✅ Residual plots
- ✅ Prediction vs Actual comparisons
- ✅ Model comparison charts

### Model Deployment

- ✅ Model persistence (save/load)
- ✅ Prediction function
- ✅ Example predictions on new data

## 📦 Installation

1. Navigate to the project directory:

```bash
cd projects/data-visualization
```

2. Install required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

Or install from root directory:

```bash
pip install -r ../../requirements.txt
```

## 💻 Usage

### Run the Main ML Script

```bash
python src/salary_prediction.py
```

The script will:

1. Generate synthetic salary dataset (100 samples)
2. Perform exploratory data analysis
3. Train multiple regression models:
   - Simple Linear Regression (Age → Salary)
   - Multiple Linear Regression (All features → Salary)
   - Polynomial Regression (Non-linear relationships)
4. Evaluate models with comprehensive metrics
5. Generate visualizations
6. Save trained models
7. Demonstrate predictions on example data

## 📈 Results

### Model Performance

**Multiple Linear Regression** (Best Model):

- **R² Score**: ~0.85-0.95 (excellent fit)
- **RMSE**: ~$5,000-$8,000
- **MAE**: ~$4,000-$6,000

**Model Comparison**:

- Multiple Linear Regression typically outperforms Simple Linear Regression
- Polynomial Regression may show slight improvement for non-linear relationships
- All models demonstrate good predictive capability

### Visualizations Generated

1. **correlation_heatmap.png**: Feature correlations
2. **feature_distributions.png**: Distribution of all features
3. **pair_plot.png**: Pairwise feature relationships
4. **regression_analysis.png**: Regression lines and predictions
5. **residual_plots.png**: Residual analysis for model validation
6. **model_comparison.png**: Side-by-side model performance comparison

## 🔧 Technical Details

### Algorithms

- **Linear Regression**: Supervised learning algorithm for regression
- **Polynomial Regression**: Extension of linear regression for non-linear relationships
- **Feature Scaling**: StandardScaler for normalization

### Preprocessing

- **Train-Test Split**: 80% training, 20% testing
- **Feature Scaling**: StandardScaler (for multiple regression)
- **Random State**: Fixed for reproducibility

### Model Evaluation

- **R² Score**: Measures proportion of variance explained
- **MSE/MAE/RMSE**: Error metrics for regression
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Residual Analysis**: Check linearity and homoscedasticity assumptions

### Feature Importance

The model learns that:

- **Education** has the strongest positive impact on salary
- **Experience** significantly increases salary
- **Age** shows positive correlation with salary

## 📁 Project Structure

```bash
data-visualization/
├── src/
│   └── salary_prediction.py      # Main ML script
├── data/
│   └── salary_data.csv          # Generated dataset
├── models/
│   ├── linear_regression_model.pkl
│   ├── polynomial_regression_model.pkl
│   ├── scaler.pkl
│   └── poly_features.pkl
├── results/
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── pair_plot.png
│   ├── regression_analysis.png
│   ├── residual_plots.png
│   └── model_comparison.png
└── README.md
```

## 🎓 Learning Outcomes

After completing this project, you will understand:

- ✅ **Supervised Learning**: Regression problems
- ✅ **Linear Regression**: Simple and multiple regression
- ✅ **Polynomial Regression**: Non-linear relationships
- ✅ **Model Evaluation**: Regression metrics (R², MSE, MAE, RMSE)
- ✅ **Feature Engineering**: Feature selection and scaling
- ✅ **Cross-Validation**: Robust model evaluation
- ✅ **Residual Analysis**: Model assumption validation
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Data Visualization**: ML-focused visualizations

## 📊 Example Predictions

The model can predict salary for new individuals:

```python
# Example: 30-year-old with 5 years experience and Bachelor's degree
Age: 30, Experience: 5, Education: Bachelor's
Predicted Salary: ~$65,000-$75,000

# Example: 45-year-old with 20 years experience and PhD
Age: 45, Experience: 20, Education: PhD
Predicted Salary: ~$120,000-$140,000
```

## 🔮 Future Improvements

- [ ] Use real-world salary dataset
- [ ] Add more features (location, industry, job title)
- [ ] Implement other regression algorithms (Ridge, Lasso, Elastic Net)
- [ ] Add feature importance visualization
- [ ] Create interactive prediction interface
- [ ] Deploy model as web API
- [ ] Add time series analysis for salary trends
- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)

## 📚 References

- scikit-learn Linear Regression documentation
- Polynomial Regression concepts
- Model evaluation metrics for regression
- Residual analysis and model assumptions

---

**Skills**: Supervised Learning, Regression, Linear Regression, Polynomial Regression, Model Evaluation, Feature Engineering, Data Visualization, Model Deployment
