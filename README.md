# Loan Default Risk Predictor (NeoRisk)

A machine learning model to predict loan default risk, deployed as a Flask web application with a modern, responsive UI.

## Features

- AI-powered loan default risk assessment
- Real-time risk prediction with probability scores
- Beautiful, modern web interface
- Bias-free machine learning model using XGBoost
- Deployed and ready for production use

## Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model (optional - model file already included):**
   ```bash
   python loan_model.py
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   - Open your browser and navigate to `http://localhost:5000`

## Deployment

### Heroku
1. Ensure you have the Heroku CLI installed
2. Create a Heroku app: `heroku create`
3. Push to Heroku: `git push heroku main`
4. The `Procfile` is already configured

### Render
1. Connect your repository to Render
2. The `render.yaml` file is already configured
3. Render will automatically detect and deploy the application

### Docker
1. Build the image: `docker build -t loan-risk-predictor .`
2. Run the container: `docker run -p 5000:5000 loan-risk-predictor`

## Model Information

The model uses XGBoost classifier with the following features:
- Annual Income
- Age
- Loan Amount
- Credit Score (300-850)
- Credit Utilization Ratio (0-1)
- Debt-to-Income Ratio (calculated)

## Requirements

- Python 3.10+
- All dependencies listed in `requirements.txt`
- `loan_model.joblib` model file (included in repository)
