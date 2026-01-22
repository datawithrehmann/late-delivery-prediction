import pandas as pd
import joblib

# 1. Load trained model
model = joblib.load("late_delivery_model.pkl")

# 2. Load new orders data
# Example file: new_orders.csv
df = pd.read_csv("new_orders.csv")

# 3. Drop ID if present (model mein use nahi hota)
if 'ID' in df.columns:
    order_ids = df['ID']
    df = df.drop(columns=['ID'])
else:
    order_ids = range(len(df))

# 4. One-hot encoding (same as training)
df = pd.get_dummies(df, drop_first=True)

# 5. Align columns with model training columns
model_features = model.feature_names_in_
df = df.reindex(columns=model_features, fill_value=0)

# 6. Make predictions
predictions = model.predict(df)

# 7. Convert prediction to business-friendly output
result = pd.DataFrame({
    "order_id": order_ids,
    "prediction": predictions
})

result["prediction_label"] = result["prediction"].map({
    0: "Late",
    1: "On-Time"
})

result["action"] = result["prediction_label"].apply(
    lambda x: "Priority Shipment" if x == "Late" else "Normal Shipment"
)

# 8. Save output
result.to_csv("prediction_results.csv", index=False)

print("Prediction completed. File saved as prediction_results.csv")
