# E-commerce Order and Product Details Dataset

## Overview

The E-commerce Order and Product Details dataset offer a comprehensive understanding of an e-commerce platform's operations. It includes detailed information about orders, items within orders, customers, payments, and products. The dataset is structured with multiple tables, each providing specific insights into different aspects of e-commerce activities.

### Orders Table
- **order_id**: Unique identifier for an order, serving as the primary key.
- **customer_id**: Unique identifier for a customer. This may not be unique across the table.
- **order_status**: Indicates the status of an order (e.g., delivered, cancelled, processing, etc.).
- **order_purchase_timestamp**: Timestamp when the customer placed the order.
- **order_approved_at**: Timestamp when the order was approved by the seller.
- **order_delivered_timestamp**: Timestamp when the order reached the customer's location.
- **order_estimated_delivery_date**: Estimated delivery date shared with the customer during order placement.

### Order Items Table
- **order_id**: Unique identifier for an order.
- **order_item_id**: Item number within each order, forming part of the primary key along with order_id.
- **product_id**: Unique identifier for a product.
- **seller_id**: Unique identifier for the seller.
- **price**: Selling price of the product.
- **shipping_charges**: Charges associated with shipping the product.

### Customers Table
- **customer_id**: Unique identifier for a customer, serving as the primary key.
- **customer_zip_code_prefix**: Customer's Zip code.
- **customer_city**: Customer's city.
- **customer_state**: Customer's state.

### Payments Table
- **order_id**: Unique identifier for an order.
- **payment_sequential**: Provides information about the payment sequence for the given order.
- **payment_type**: Type of payment (e.g., credit_card, debit_card, etc.).
- **payment_installments**: Payment installment number for credit cards.
- **payment_value**: Transaction value.

### Products Table
- **product_id**: Unique identifier for each product, serving as the primary key.
- **product_category_name**: Name of the category to which the product belongs.
- **product_weight_g**: Product weight in grams.
- **product_length_cm**: Product length in centimeters.
- **product_height_cm**: Product height in centimeters.
- **product_width_cm**: Product width in centimeters.

## Dataset Split

### Train Set
- df_orders.csv
- df_order_items.csv
- df_customers.csv
- df_payments.csv
- df_products.csv

### Test Set
- df_orders.csv
- df_order_items.csv
- df_customers.csv
- df_payments.csv
- df_products.csv

Note: Ensure that the features for both the train and test data are consistent. For example:

- Train features: [A, B, C, D, E, F, G, H] with the target column [is_late]
- Test features: [A, B, C, D, E, F, G, H]

## Submission

- sample_submission.csv: An example file for submission to Kaggle. Ensure that each team pays attention to the order_id for joining with other relevant data during the submission process.

# Machine Learning Competition README

## Overview

This document outlines the rules and guidelines for the Machine Learning competition in the context of supervised learning and data management.

### Competition Type

The competition follows a supervised learning approach, where participants are tasked with building a model based on labeled data.

### Data Management

Five data files are provided for training purposes. Participants are responsible for labeling the data according to the given task and ensuring label quality.

### Training and Testing Process

Participants are required to train a machine learning model using the labeled training data. After training, testing should be conducted on the provided data to evaluate the model's performance.

### Results Submission

Participants must submit their results in the specified format, following the structure demonstrated in the provided sample_submission example.

### Final Evaluation

#### Evaluation Metric

The primary evaluation metric for this competition is accuracy. Accuracy is calculated by dividing the number of correct predictions by the total number of predictions using the formula:

\[ \text{accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}} \]

## Code and Method Explanation

To address the specific problem presented in this competition, the following steps were taken in the implementation:

1. **Data Preprocessing:**
   - Conversion of relevant columns to datetime format:

     ```python
     result['order_delivered_timestamp'] = pd.to_datetime(result['order_delivered_timestamp'])
     result['order_estimated_delivery_date'] = pd.to_datetime(result['order_estimated_delivery_date'])
     result['is_late'] = (result['order_delivered_timestamp'] > result['order_estimated_delivery_date']).astype(int)
     ```

     This step ensures that time-related columns are in the proper datetime format.

2. **Feature Selection:**
   - Selection of relevant features:

     ```python
     features = ['customer_zip_code_prefix', 'price', 'shipping_charges',
                 'payment_sequential', 'payment_installments', 'payment_value',
                 'product_weight_g', 'product_length_cm', 'product_height_cm',
                 'product_width_cm']
     X = result[features]
     y = result['is_late']
     ```

     Identified features crucial for predicting the target variable 'is_late'.

3. **Handling Missing Values:**
   - Utilization of SimpleImputer:

     ```python
     imputer = SimpleImputer(strategy='mean')
     X = imputer.fit_transform(X)
     ```

     Missing values in selected features were handled using the SimpleImputer from scikit-learn with a mean imputation strategy.

4. **Model Training:**
   - Implementation of RandomForestClassifier:

     ```python
     clf = RandomForestClassifier(random_state=42)
     clf.fit(X, y)
     ```

     A RandomForestClassifier was chosen for its effectiveness in classification tasks. The classifier was trained using the labeled training data ('X' features and 'y' target variable).

This process ensures that the model is trained on relevant features, accounting for missing values, and utilizing a RandomForestClassifier for the classification task. The resulting model is then ready for testing and evaluation on new data.

## Preview

![Model RandomForestClassifier](https://github.com/boycakra/Ecommerce-Laptop/assets/48791469/b0910737-8bf5-407a-9d71-1772331adb58)

![Docky_test](https://github.com/boycakra/Ecommerce-Laptop/assets/48791469/8294d31b-70cc-4991-89ef-dc32e98d344c)
