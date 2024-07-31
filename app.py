import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
                            'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 
                            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age', 
                            'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing required features"}), 400

        input_data = pd.DataFrame([data]) 

       

        prediction = model.predict(input_data)

        return jsonify({"Credit_Score": prediction[0]})  

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
