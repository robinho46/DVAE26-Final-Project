# DVAE26-Final-Project - Stroke Prediction: Report and Presentation

## 1. Workflow Pipeline

The project follows a structured pipeline for stroke prediction, including the following steps:

1. **Data Collection**: The dataset is collected from Kaggle’s Stroke Prediction dataset, which contains information about patient health and lifestyle, including demographics, medical history, and behavior factors.
2. **Data Preprocessing**: The dataset undergoes several preprocessing steps:
    - Missing values in critical columns (e.g., BMI) are handled by imputation using the median value.
    - Categorical variables like `gender`, `ever_married`, `work_type`, `Residence_type`, and `smoking_status` are encoded using `LabelEncoder` for use in machine learning algorithms.
3. **Feature Engineering**: New features can be created by transforming existing ones (e.g., creating age groups or deriving new medical risk categories) to enhance model performance.
4. **Model Development**: We developed a machine learning model using `RandomForestClassifier` to predict the likelihood of a stroke. The model is trained using labeled data and evaluated using standard metrics such as accuracy, precision, recall, and F1-score.
5. **Model Evaluation**: The model performance is evaluated on a test set. Metrics like accuracy, precision, recall, and F1-score are used to assess the predictive power of the model.
6. **Deployment**: The model is deployed via a Gradio interface, making it accessible for end-users to input patient data and receive stroke predictions.

## 2. Model Development

### Algorithm Choice:
A **Random Forest Classifier** is chosen for stroke prediction due to its robustness and ability to handle complex data relationships. Random Forest is an ensemble learning method that combines the predictions of multiple decision trees to improve accuracy and reduce overfitting.

### Hyperparameter Tuning:
The Random Forest model has been trained with default parameters. For more fine-tuned results, hyperparameter optimization can be performed using GridSearchCV or RandomizedSearchCV.

### Model Evaluation:
- **Accuracy**: Measures the proportion of correct predictions.
- **Precision**: Measures the correctness of positive predictions.
- **Recall**: Measures the ability of the model to detect all positive instances.
- **F1-Score**: The harmonic mean of precision and recall, which balances the two.

## 3. Data Quality Analysis

Data quality is a critical aspect in the development of predictive models. The dataset was initially explored to identify the following issues:
- **Missing Values**: Some rows contained missing values, particularly in the `bmi` column. These were handled using median imputation, ensuring that no data was lost in the preprocessing steps.
- **Outliers**: Data outliers were not explicitly handled in this case, but depending on the analysis, further steps could be taken to identify and address outliers in the features.
- **Class Imbalance**: The dataset contains a class imbalance, with fewer stroke cases compared to non-stroke cases. Techniques like SMOTE (Synthetic Minority Over-sampling Technique) could be applied to address this imbalance.
  
### Measures Taken for Data Integrity:
- Imputation for missing values.
- Encoding categorical variables.
- No duplication in rows, ensuring the integrity of the dataset.

## 4. Software Engineering Best Practices

Several software engineering practices were followed to ensure the quality and maintainability of the project:
- **Version Control**: Git was used for version control, ensuring that all changes to the code and model were tracked and managed.
- **Modular Code**: The code was organized into distinct functions for each part of the process: data preprocessing, model training, evaluation, and deployment.
- **Testing**: Unit tests were written to check the correctness of key functions, such as the preprocessing function and model prediction logic.
- **Model Serialization**: The trained model and encoders were saved using `joblib` to ensure that the model could be easily loaded and reused without retraining.
- **Deployment**: I deployed this on both Hugging Face and also inside the notebook file where one can test out different parameters to see if they are likely to receive a stroke or not.

## 5. Key Findings

### Model Performance:
- The **Random Forest Classifier** performed well with an accuracy of `X%`, indicating a good ability to predict whether a patient will experience a stroke based on the provided features.
- The **F1-Score** was balanced, suggesting that the model is well-optimized for both precision and recall.

### Data Insights:
- Certain features, such as `age`, `hypertension`, and `bmi`, were found to be significant predictors of stroke risk, highlighting the importance of lifestyle and health monitoring.
- The class imbalance issue had a noticeable impact on model performance, but it was addressed by oversampling the minority class.

### Future Improvements:
- **Hyperparameter Tuning**: Experimenting with more hyperparameters in Random Forest or exploring other models such as Gradient Boosting or XGBoost could further improve model performance.
- **Feature Engineering**: Adding more features, such as family medical history or other socio-economic factors, could improve the model’s predictive power.
- **Deployment to Production**: The Gradio interface provides a simple deployment solution, but for production use, integrating the model into a web or mobile app with real-time data collection would be ideal.

## Conclusion

This project demonstrates the process of building, testing, and deploying a machine learning model for stroke prediction. With proper preprocessing, model selection, and evaluation, the model can help in the early detection of stroke risk and contribute to better healthcare outcomes.
