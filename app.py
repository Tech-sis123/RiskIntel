from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os
import sys
import numpy as np

app = Flask(__name__)

# Debug: Print environment info
print("=" * 60)
print("LOAN DEFAULT RISK PREDICTOR - APPLICATION STARTING")
print("=" * 60)
print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# Load model with enhanced error handling
model = None
feature_info = None

# Try multiple possible paths for the model file
possible_paths = [
    'loan_model.joblib',
    os.path.join(os.path.dirname(__file__), 'loan_model.joblib'),
    os.path.join(os.getcwd(), 'loan_model.joblib')
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path:
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from {model_path}!")
        
        # Try to load feature information
        feature_info_path = model_path.replace('loan_model.joblib', 'model_features.joblib')
        if os.path.exists(feature_info_path):
            feature_info = joblib.load(feature_info_path)
            print(f"✅ Feature information loaded: {len(feature_info.get('feature_names', []))} features")
        else:
            # Default feature order (matching loan_model.py)
            feature_info = {
                'feature_names': [
                    'income', 'age', 'loan_amount', 'credit_score', 
                    'debt_to_income', 'credit_utilization',
                    'loan_to_income_ratio', 'credit_score_category',
                    'income_adequacy', 'age_category', 'risk_score'
                ],
                'feature_order': [
                    'income', 'age', 'loan_amount', 'credit_score', 
                    'debt_to_income', 'credit_utilization',
                    'loan_to_income_ratio', 'credit_score_category',
                    'income_adequacy', 'age_category', 'risk_score'
                ]
            }
            print(f"⚠️  Using default feature order: {len(feature_info['feature_names'])} features")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        model = None
else:
    print(f"⚠️  Model file not found in any of these locations: {possible_paths}")
    print("Please ensure loan_model.joblib exists in the project root")

def calculate_credit_score_category(credit_score):
    """Categorize credit score: Poor, Fair, Good, Excellent"""
    if credit_score < 580:
        return 0  # Poor
    elif credit_score < 670:
        return 1  # Fair
    elif credit_score < 740:
        return 2  # Good
    else:
        return 3  # Excellent

def calculate_age_category(age):
    """Categorize age into risk groups"""
    if age < 25:
        return 0  # Very Young
    elif age < 35:
        return 1  # Young
    elif age < 50:
        return 2  # Middle
    elif age < 65:
        return 3  # Mature
    else:
        return 4  # Senior

def calculate_risk_score(credit_score, debt_to_income, credit_utilization):
    """Calculate composite risk score"""
    credit_risk = (850 - credit_score) / 550 * 0.4
    dti_risk = min(debt_to_income / 0.5, 1.0) * 0.3
    util_risk = credit_utilization * 0.3
    return credit_risk + dti_risk + util_risk

def calculate_bill_payment_bonus(bill_payment):
    """Calculate bonus points based on bill payment history"""
    bonuses = {
        'always': 20,      # Always on time = +20 points
        'mostly': 10,      # Mostly on time = +10 points
        'often_late': -10, # Often late = -10 points
        'prefer_not': 0,   # Prefer not to say = 0 points
        '': 0              # Not provided = 0 points
    }
    return bonuses.get(bill_payment, 0)

def calculate_reference_bonus(ref1_name, ref1_phone, ref2_name, ref2_phone):
    """Calculate bonus points based on character references"""
    ref_count = 0
    if ref1_name and ref1_phone:
        ref_count += 1
    if ref2_name and ref2_phone:
        ref_count += 1
    
    if ref_count == 2:
        return 15  # 2 references = +15 points
    elif ref_count == 1:
        return 5   # 1 reference = +5 points
    else:
        return 0   # 0 references = 0 points

def calculate_habits_bonus(financial_habits):
    """Calculate bonus points based on financial habits"""
    # Each good habit = +10 points
    habit_points = {
        'savings_habit': 10,
        'no_gambling': 10,
        'track_spending': 10,
        'emergency_fund': 10  # Emergency savings fund
    }
    
    total_bonus = 0
    for habit in financial_habits:
        total_bonus += habit_points.get(habit, 0)
    
    return total_bonus

def calculate_cash_flow_score(monthly_income, monthly_expenses, monthly_savings):
    """Calculate cash flow score based on disposable income"""
    if monthly_income <= 0:
        return 0
    
    disposable_income = monthly_income - monthly_expenses
    
    # Positive disposable income is good
    if disposable_income > 0:
        # Calculate as percentage of income
        disposable_ratio = disposable_income / monthly_income
        # Convert to points (0-30 points)
        return min(disposable_ratio * 30, 30)
    else:
        # Negative disposable income is bad
        return -10

def assess_application_safety(probability, credit_score, debt_to_income, income, loan_amount):
    """
    Comprehensive banking-style risk assessment
    Returns: (is_safe, risk_level, recommendation, explanation)
    """
    default_prob = probability
    risk_percent = default_prob * 100
    
    # Banking industry risk thresholds
    if risk_percent < 20:
        risk_level = "Low Risk"
        is_safe = True
        recommendation = "APPROVE - Safe Application"
    elif risk_percent < 35:
        risk_level = "Moderate Risk"
        is_safe = True
        recommendation = "APPROVE with Conditions - Monitor closely"
    elif risk_percent < 50:
        risk_level = "Medium-High Risk"
        is_safe = False
        recommendation = "REVIEW REQUIRED - Additional documentation needed"
    else:
        risk_level = "High Risk"
        is_safe = False
        recommendation = "DECLINE - High default probability"
    
    # Build detailed explanation
    explanation_parts = []
    
    # Credit score analysis
    if credit_score >= 740:
        explanation_parts.append("Excellent credit score indicates strong creditworthiness.")
    elif credit_score >= 670:
        explanation_parts.append("Good credit score shows reliable payment history.")
    elif credit_score >= 580:
        explanation_parts.append("Fair credit score suggests some credit concerns.")
    else:
        explanation_parts.append("Poor credit score indicates significant credit risk.")
    
    # Debt-to-income analysis
    if debt_to_income < 0.2:
        explanation_parts.append("Low debt-to-income ratio shows strong repayment capacity.")
    elif debt_to_income < 0.36:
        explanation_parts.append("Moderate debt-to-income ratio is within acceptable range.")
    elif debt_to_income < 0.5:
        explanation_parts.append("High debt-to-income ratio may strain repayment ability.")
    else:
        explanation_parts.append("Very high debt-to-income ratio poses significant repayment risk.")
    
    # Overall assessment
    if is_safe:
        explanation_parts.append(f"Overall assessment: This application appears SAFE with a {risk_percent:.1f}% default probability. The applicant demonstrates acceptable risk factors.")
    else:
        explanation_parts.append(f"Overall assessment: This application presents RISK with a {risk_percent:.1f}% default probability. The applicant shows concerning risk indicators.")
    
    explanation = " ".join(explanation_parts)
    
    return is_safe, risk_level, recommendation, explanation

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Check if model is loaded
            if model is None:
                return jsonify({'error': 'Model not available. Please ensure loan_model.joblib exists.'}), 500
            
            print("\n" + "=" * 60)
            print("NEW LOAN APPLICATION RECEIVED")
            print("=" * 60)
            print("📦 Received form data:", dict(request.form))
            
            # Parse and validate inputs
            try:
                income = float(request.form['income'])
                age = float(request.form['age'])
                loan_amount = float(request.form['loan_amount'])
                credit_score = float(request.form['credit_score'])
                credit_utilization = float(request.form.get('credit_utilization', 0.5))
                
                # New enhanced fields
                monthly_income = float(request.form.get('monthly_income', 0) or 0)
                monthly_expenses = float(request.form.get('monthly_expenses', 0) or 0)
                monthly_savings = float(request.form.get('monthly_savings', 0) or 0)
                bill_payment = request.form.get('bill_payment', '')
                
                # Character references
                ref1_name = request.form.get('ref1_name', '').strip()
                ref1_phone = request.form.get('ref1_phone', '').strip()
                ref2_name = request.form.get('ref2_name', '').strip()
                ref2_phone = request.form.get('ref2_phone', '').strip()
                
                # Financial habits (checkboxes come as list)
                financial_habits = request.form.getlist('financial_habits')
                
            except (KeyError, ValueError) as e:
                return jsonify({'error': f'Invalid input: {str(e)}. Please check all fields are filled correctly.'}), 400
            
            # Basic validation - accept any realistic banking industry values
            # Only validate for impossible/illegal values, let the AI model assess risk
            
            if income <= 0:
                return jsonify({'error': 'Annual income must be greater than $0. Please enter a valid income amount.'}), 400
            if loan_amount <= 0:
                return jsonify({'error': 'Loan amount must be greater than $0. Please enter a valid loan amount.'}), 400
            if credit_score < 300 or credit_score > 850:
                return jsonify({'error': 'Credit score must be between 300-850 (standard FICO range). Please enter a valid credit score.'}), 400
            if credit_utilization < 0 or credit_utilization > 1:
                return jsonify({'error': 'Credit utilization must be between 0-1 (0% to 100%). For example, 0.5 means 50% utilization.'}), 400
            if age < 18:
                return jsonify({'error': 'Age must be at least 18 (legal age for loan applications).'}), 400
            if age > 120:
                return jsonify({'error': 'Please enter a valid age.'}), 400
            
            # Note: We don't restrict income/loan amounts or ratios
            # The AI model will assess risk based on these values
            # Banks accept applications with various income levels and loan amounts
            
            # Calculate derived features (matching model training)
            debt_to_income = loan_amount / income
            loan_to_income_ratio = loan_amount / income
            credit_score_category = calculate_credit_score_category(credit_score)
            age_category = calculate_age_category(age)
            income_adequacy = income / (loan_amount + 1)
            risk_score = calculate_risk_score(credit_score, debt_to_income, credit_utilization)
            
            # Calculate enhanced bonuses
            bill_payment_bonus = calculate_bill_payment_bonus(bill_payment)
            reference_bonus = calculate_reference_bonus(ref1_name, ref1_phone, ref2_name, ref2_phone)
            habits_bonus = calculate_habits_bonus(financial_habits)
            cash_flow_bonus = calculate_cash_flow_score(monthly_income, monthly_expenses, monthly_savings)
            
            total_bonus = bill_payment_bonus + reference_bonus + habits_bonus + cash_flow_bonus
            
            print(f" Enhanced Scoring:")
            print(f"   Bill Payment Bonus: {bill_payment_bonus} points")
            print(f"   Reference Bonus: {reference_bonus} points")
            print(f"   Habits Bonus: {habits_bonus} points")
            print(f"   Cash Flow Bonus: {cash_flow_bonus:.1f} points")
            print(f"   Total Bonus: {total_bonus:.1f} points")
            
            # Create input dictionary with all features
            input_data = {
                'income': income,
                'age': age,
                'loan_amount': loan_amount,
                'credit_score': credit_score,
                'debt_to_income': debt_to_income,
                'credit_utilization': credit_utilization,
                'loan_to_income_ratio': loan_to_income_ratio,
                'credit_score_category': credit_score_category,
                'income_adequacy': income_adequacy,
                'age_category': age_category,
                'risk_score': risk_score
            }
            
            print("🔢 Processed inputs:")
            for key, value in input_data.items():
                print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
            
            # Get feature order from model or use default
            if feature_info and 'feature_order' in feature_info:
                feature_order = feature_info['feature_order']
            else:
                feature_order = list(input_data.keys())
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])[feature_order]
            
            # Ensure no NaN or infinite values
            input_df = input_df.replace([np.inf, -np.inf], np.nan)
            if input_df.isnull().any().any():
                # Fill with median if available, otherwise 0
                input_df = input_df.fillna(0)
            
            print("📊 Input DataFrame shape:", input_df.shape)
            print("📊 Feature order:", feature_order)
            
            # Make prediction
            try:
                # Get probability of default (class 1)
                base_probability = model.predict_proba(input_df)[0][1]
                base_risk_percent = float(base_probability * 100)
                
                print(f"🎯 Base prediction probability: {base_probability:.4f} ({base_risk_percent:.1f}%)")
                
                # Adjust risk based on enhanced bonuses
                # Convert bonus points to risk reduction (negative points reduce risk)
                # 1 point = 0.5% risk reduction, max reduction of 20%
                risk_adjustment = -min(total_bonus * 0.5, 20)
                adjusted_risk_percent = max(0, min(100, base_risk_percent + risk_adjustment))
                adjusted_probability = adjusted_risk_percent / 100
                
                print(f"📊 Adjusted risk: {adjusted_risk_percent:.1f}% (adjusted by {risk_adjustment:.1f}%)")
                
                # Use adjusted risk for final assessment
                probability = adjusted_probability
                risk_percent = adjusted_risk_percent
                
                # Comprehensive risk assessment
                is_safe, risk_level, recommendation, explanation = assess_application_safety(
                    probability, credit_score, debt_to_income, income, loan_amount
                )
                
                # Add bonus information to explanation
                if total_bonus > 0:
                    explanation += f" Enhanced scoring factors improved the applicant's risk profile by {abs(risk_adjustment):.1f}%."
                elif total_bonus < 0:
                    explanation += f" Some factors increased the applicant's risk profile by {abs(risk_adjustment):.1f}%."
                
                print(f"✅ Assessment: {risk_level} - {recommendation}")
                print(f"   Safe: {is_safe}")
                
                # Determine risk category for UI
                if risk_percent >= 50:
                    risk_category = "High Risk"
                    risk_class = "high-risk"
                elif risk_percent >= 35:
                    risk_category = "Medium-High Risk"
                    risk_class = "medium-high-risk"
                elif risk_percent >= 20:
                    risk_category = "Moderate Risk"
                    risk_class = "moderate-risk"
                else:
                    risk_category = "Low Risk"
                    risk_class = "low-risk"
                
                return jsonify({
                    'prediction': float(probability),
                    'risk_percent': round(risk_percent, 2),
                    'base_risk_percent': round(base_risk_percent, 2),
                    'risk_adjustment': round(risk_adjustment, 2),
                    'risk_category': risk_category,
                    'risk_class': risk_class,
                    'is_safe': is_safe,
                    'recommendation': recommendation,
                    'explanation': explanation,
                    'probability': f"{risk_percent:.1f}%",
                    'details': {
                        'credit_score_category': ['Poor', 'Fair', 'Good', 'Excellent'][credit_score_category],
                        'debt_to_income_ratio': round(debt_to_income, 3),
                        'credit_utilization': round(credit_utilization * 100, 1),
                        'bill_payment_bonus': bill_payment_bonus,
                        'reference_bonus': reference_bonus,
                        'habits_bonus': habits_bonus,
                        'cash_flow_bonus': round(cash_flow_bonus, 1),
                        'total_bonus': round(total_bonus, 1)
                    }
                })
                
            except Exception as pred_error:
                print(f"❌ Prediction error: {str(pred_error)}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'Prediction failed: {str(pred_error)}. Please check that all inputs are valid numbers.'
                }), 500
            
        except Exception as e:
            print(f"🔥 Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Server error: {str(e)}'}), 500
    
    # GET request - render the HTML page
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Only enable debug mode in development, not in production
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    print(f"\n🚀 Starting server on port {port} (debug={debug_mode})")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
