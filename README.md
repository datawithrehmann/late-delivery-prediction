# Late Delivery Prediction (E-commerce)

## Problem
Predict whether an order will be delivered late or on time before shipment.

## Business Impact
Late deliveries reduce customer trust and ratings. Early prediction helps prioritize high-risk orders.

## Approach
- Supervised binary classification
- Logistic Regression with class weight tuning
- Focus on late delivery recall

## Dataset
E-commerce shipment dataset with order, product, and logistics features.

## How to Run
1. Train model in notebook
2. Save model as `late_delivery_model.pkl`
3. Add new orders in `new_orders.csv`
4. Run:
   ```bash
   python predict.py
