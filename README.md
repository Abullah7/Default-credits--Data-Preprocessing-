# Default Credit Classification

This repository contains a machine learning project for predicting credit defaults using a classification model. The project utilizes the "Default of Credit Card Clients" dataset from Kaggle to demonstrate data preprocessing, model training, evaluation, and prediction. The primary objective is to classify whether a credit customer will default or not.

## Dataset

The dataset used in this project is the [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) from Kaggle. It includes various features related to credit customers and their financial behavior. The target variable indicates whether the customer has defaulted on their credit (1 for default, 0 for no default).

## Project Overview

The notebook  covers the following:

1. **Data Loading**: Importing and exploring the dataset.
2. **Data Preprocessing**: Cleaning the data, handling missing values, encoding categorical variables, and feature scaling.
3. **Exploratory Data Analysis (EDA)**: Visualizing data distributions, correlations, and relationships between features.
4. **Model Training**: Training various machine learning models to predict credit default.
5. **Model Evaluation**: Evaluating model performance using metrics like accuracy, precision, recall, and F1-score.
6. **Prediction**: Making predictions on new data and validating model performance.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Abdullah7/default-credit preprocessing.git
    cd default-credit-classification
    ```

2. **Install dependencies**:

    Ensure you have Python installed, then install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the notebook:

```bash
jupyter notebook "model_3.ipynb"
```

### Notebook Sections

- **Loading the Dataset**: Importing the "Default of Credit Card Clients" dataset and displaying initial data.
- **Data Preprocessing**: Cleaning and preparing the data for training, including handling missing values, encoding categorical features, and scaling numerical features.
- **Exploratory Data Analysis (EDA)**: Visualizing the data to understand distributions, correlations, and feature importance.
- **Model Training**: Training multiple classification models such as Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVM).
- **Model Evaluation**: Evaluating model performance using metrics such as accuracy, precision, recall, and F1-score. This section includes confusion matrices and ROC curves to provide a comprehensive evaluation.
- **Prediction**: Using the trained model to make predictions on new data and evaluating the predictions' accuracy.

## Results

The notebook demonstrates the process of training and evaluating various classification models. The best-performing model is selected based on evaluation metrics, and its performance is highlighted in the results.

### Key Metrics

- **Accuracy**: Measure of the model's overall correctness.
- **Precision**: Measure of the correctness of positive predictions.
- **Recall**: Measure of the model's ability to identify positive instances.
- **F1-Score**: Harmonic mean of precision and recall, providing a single metric to evaluate the model's performance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have suggestions or bug reports.

## License

This project is licensed under the MIT License.

## Acknowledgements

- The dataset used in this project is the [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) from Kaggle.
- Special thanks to the contributors of the dataset and the open-source community.

---

Feel free to reach out with any questions or suggestions!
