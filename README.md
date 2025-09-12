# BigMart-Sales-Prediction

This project presents a complete end-to-end machine learning solution for predicting item sales at various BigMart outlets. The solution encompasses data loading, a robust machine learning pipeline, and a user-friendly web application for real-time sales predictions.

App live link:=> https://bigmart-sales-prediction-euwesqnogzruwunhjsptxv.streamlit.app/

Key Features
End-to-End ML Pipeline: A structured workflow that handles data from ingestion to model deployment.

Data Management: Raw data (in XML format) is ingested and loaded into a MySQL database for structured storage and easy access.

Model Training: A Jupyter Notebook explores and trains multiple regression models, saving the best-performing one to a pickle file.

Best Model Selection: The Gradient Boosting Regressor was selected as the final model due to its superior performance, as validated by an R² score of 0.5932.

Interactive Web App: A clean and intuitive web application built with Streamlit allows users to input item and outlet details and receive instant sales predictions.

Project Files
app.py: The main script for the Streamlit web application. It loads the trained model and provides an interactive interface for sales prediction.

bigmart_best_model.pkl: The serialized machine learning pipeline, containing the pre-processing steps and the trained GradientBoosting model.

df_item.xml: The raw dataset containing detailed information about various items.

df_outlet.xml: The raw dataset containing detailed information about different outlets.

df_sales.xml: The raw dataset containing the target sales data for each item-outlet combination.

load_data.ipynb: A Jupyter Notebook that demonstrates how to connect to a MySQL server and load the XML data into the database.

train_ml_pkl.ipynb: A Jupyter Notebook that reads data from MySQL, trains several machine learning models, and saves the best-performing model to bigmart_best_model.pkl.

Technologies Used
Python: The core programming language.

Scikit-learn: For building and evaluating the machine learning models.

Pandas: For data manipulation and analysis.

Streamlit: For building the interactive web application.

MySQL: For persistent data storage.

Jupyter Notebook: For exploratory data analysis and model training.

Pickle: For serializing the trained model.

How to Run
Set up MySQL: Ensure you have a MySQL server running locally.

Load Data: Execute the load_data.ipynb notebook to create the database and load the XML files.

Train Model: Run the train_ml_pkl.ipynb notebook to train the machine learning model and save it.

Launch App: Run the Streamlit application from your terminal using streamlit run app.py.
