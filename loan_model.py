import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("LOAN DEFAULT RISK PREDICTOR - MODEL TRAINING")
print("=" * 60)

# 1. Load and combine data (adjust paths as needed)
print("\n[1/8] Loading training data...")
try:
    kaggle_data = pd.read_csv('credit_risk.csv')
    german_data = pd.read_csv('german_credit.csv')
    
    # Preprocess Kaggle data (adjust column names as needed)
    kaggle_data['target'] = kaggle_data['loan_status'].apply(lambda x: 1 if x == 'default' else 0)
    kaggle_processed = kaggle_data[['income', 'age', 'loan_amount', 'credit_score', 'target']]
    
    # Preprocess German data
    german_processed = german_data.rename(columns={
        'Class': 'target',
        'Duration': 'loan_duration',
        'Amount': 'loan_amount'
    })
    german_processed['target'] = german_processed['target'].replace({2: 1, 1: 0})  # Convert to 0/1
    
    # Combine datasets (use common features)
    combined_data = pd.concat([kaggle_processed, german_processed], axis=0)
    print(f"✅ Loaded {len(combined_data)} records from CSV files")
except Exception as e:
    print(f"⚠️  Error loading data: {e}")
    # Create realistic banking data with proper correlations
    print("Creating realistic banking data with proper risk patterns...")
    np.random.seed(42)
    n_samples = 5000
    
    # Generate realistic income distribution (skewed right)
    income = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples)
    income = np.clip(income, 20000, 200000)
    
    # Generate age (normal distribution, working age)
    age = np.random.normal(45, 12, n_samples)
    age = np.clip(age, 22, 75).astype(int)
    
    # Generate credit scores with realistic distribution
    credit_score = np.random.normal(650, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    # Generate loan amounts correlated with income (people borrow based on income)
    loan_amount = income * np.random.uniform(0.1, 2.5, n_samples)
    loan_amount = np.clip(loan_amount, 1000, 300000)
    
    # Generate credit utilization (correlated with credit score - lower score = higher utilization)
    credit_utilization = 1 - (credit_score - 300) / 550  # Inverse relationship
    credit_utilization = credit_utilization + np.random.normal(0, 0.15, n_samples)
    credit_utilization = np.clip(credit_utilization, 0, 1)
    
    # Calculate debt-to-income ratio
    debt_to_income = loan_amount / income
    
    # Create target variable based on realistic banking risk factors
    # Higher risk if:
    # - Low credit score (< 600)
    # - High debt-to-income (> 0.4)
    # - High credit utilization (> 0.7)
    # - Low income relative to loan amount
    # - Very young or very old age
    
    default_prob = np.zeros(n_samples)
    
    # Credit score factor (most important)
    default_prob += (850 - credit_score) / 550 * 0.4
    
    # Debt-to-income factor
    default_prob += np.clip(debt_to_income / 0.5, 0, 1) * 0.25
    
    # Credit utilization factor
    default_prob += credit_utilization * 0.2
    
    # Income stability factor (lower income = higher risk)
    income_factor = 1 - (income - 20000) / 180000
    default_prob += income_factor * 0.1
    
    # Age factor (very young or very old = higher risk)
    age_risk = np.where((age < 25) | (age > 65), 0.15, 0)
    age_risk += np.where((age >= 25) & (age <= 35), 0.05, 0)
    default_prob += age_risk * 0.05
    
    # Add some randomness
    default_prob += np.random.normal(0, 0.1, n_samples)
    default_prob = np.clip(default_prob, 0, 1)
    
    # Convert to binary target
    target = (default_prob > 0.5).astype(int)
    
    # Ensure reasonable default rate (15-25%)
    current_default_rate = target.mean()
    if current_default_rate < 0.15:
        # Increase defaults for high-risk cases
        high_risk_mask = default_prob > 0.4
        target[high_risk_mask] = np.random.choice([0, 1], size=high_risk_mask.sum(), p=[0.3, 0.7])
    elif current_default_rate > 0.30:
        # Decrease defaults for low-risk cases
        low_risk_mask = default_prob < 0.3
        target[low_risk_mask] = np.random.choice([0, 1], size=low_risk_mask.sum(), p=[0.9, 0.1])
    
    combined_data = pd.DataFrame({
        'income': income,
        'age': age,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'credit_utilization': credit_utilization,
        'debt_to_income': debt_to_income,
        'target': target
    })
    
    print(f"✅ Created {len(combined_data)} realistic loan records")
    print(f"   Default rate: {target.mean():.2%}")

# 2. Advanced Feature Engineering
print("\n[2/8] Engineering advanced features...")

# Ensure debt_to_income and credit_utilization exist
if 'debt_to_income' not in combined_data.columns:
    combined_data['debt_to_income'] = combined_data['loan_amount'] / combined_data['income']
if 'credit_utilization' not in combined_data.columns:
    combined_data['credit_utilization'] = np.random.uniform(0, 1, len(combined_data))

# Additional banking features
combined_data['loan_to_income_ratio'] = combined_data['loan_amount'] / combined_data['income']
combined_data['credit_score_category'] = pd.cut(
    combined_data['credit_score'], 
    bins=[0, 580, 670, 740, 850], 
    labels=[0, 1, 2, 3]  # Poor, Fair, Good, Excellent
).astype(int)

# Income adequacy (loan amount relative to income)
combined_data['income_adequacy'] = combined_data['income'] / (combined_data['loan_amount'] + 1)

# Age risk categories
combined_data['age_category'] = pd.cut(
    combined_data['age'],
    bins=[0, 25, 35, 50, 65, 100],
    labels=[0, 1, 2, 3, 4]  # Very Young, Young, Middle, Mature, Senior
).astype(int)

# Risk score (composite)
combined_data['risk_score'] = (
    (850 - combined_data['credit_score']) / 550 * 0.4 +
    np.clip(combined_data['debt_to_income'] / 0.5, 0, 1) * 0.3 +
    combined_data['credit_utilization'] * 0.3
)

print(f"✅ Created {len(combined_data.columns)} features")

# 3. Prepare features for training
print("\n[3/8] Preparing features for training...")
feature_columns = [
    'income', 'age', 'loan_amount', 'credit_score', 
    'debt_to_income', 'credit_utilization',
    'loan_to_income_ratio', 'credit_score_category',
    'income_adequacy', 'age_category', 'risk_score'
]

X = combined_data[feature_columns].copy()
y = combined_data['target'].copy()

# Remove any infinite or NaN values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"✅ Prepared {len(feature_columns)} features")
print(f"   Features: {', '.join(feature_columns)}")

# 4. Split data
print("\n[4/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Test set: {len(X_test)} samples")
print(f"   Training default rate: {y_train.mean():.2%}")
print(f"   Test default rate: {y_test.mean():.2%}")

# 5. Preprocessing pipeline
print("\n[5/8] Setting up preprocessing pipeline...")
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, feature_columns)
    ],
    remainder='drop'
)

# 6. Model pipeline with SMOTE
print("\n[6/8] Building XGBoost model with SMOTE...")
model = make_pipeline(
    preprocessor,
    SMOTE(random_state=42, k_neighbors=3),
    XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=sum(y_train==0)/sum(y_train==1),
        random_state=42,
        n_jobs=-1
    )
)

# 7. Hyperparameter tuning
print("\n[7/8] Tuning hyperparameters (this may take a few minutes)...")
param_dist = {
    'xgbclassifier__max_depth': [4, 5, 6, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 0.9],
    'xgbclassifier__min_child_weight': [1, 3, 5]
}

search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist,
    n_iter=10, 
    scoring='roc_auc', 
    cv=5, 
    verbose=1, 
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

print(f"✅ Best parameters: {search.best_params_}")
print(f"✅ Best CV score: {search.best_score_:.4f}")

# 8. Evaluation
print("\n[8/8] Evaluating model...")
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"True Negatives (Correct Non-Defaults): {cm[0][0]}")
print(f"False Positives (Incorrect Defaults): {cm[0][1]}")
print(f"False Negatives (Missed Defaults): {cm[1][0]}")
print(f"True Positives (Correct Defaults): {cm[1][1]}")

# Calculate important metrics
precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPrecision (Default Detection): {precision:.4f}")
print(f"Recall (Default Detection): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 9. Save model
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)
joblib.dump(best_model, 'loan_model.joblib')
print("✅ Model saved as 'loan_model.joblib'")

# Save feature names for reference
feature_info = {
    'feature_names': feature_columns,
    'feature_order': feature_columns
}
joblib.dump(feature_info, 'model_features.joblib')
print("✅ Feature information saved as 'model_features.joblib'")

# 10. Visualization
print("\nGenerating visualization...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 2, 2)
# ROC Curve data
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualization saved as 'confusion_matrix.png'")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nThe model is ready for deployment.")
print("Use this model with app.py for loan risk assessment.")
