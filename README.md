# Predicting Smoker Status using Machine Learning :smoking:

This project aims to predict the smoker status of individuals using machine learning models. The dataset used is sourced from a Kaggle competition. The project involves data cleaning, model creation, hyperparameter tuning, and model validation.

## :page_with_curl: Dataset

The dataset contains various health attributes of individuals such as age, weight, height, and more. The target variable is the 'smoker' status, which is a binary variable indicating whether the individual is a smoker or not. 

## :wrench: Dependencies

The project is implemented in Python, and uses the following libraries:

- pandas
- numpy
- matplotlib
- scikit-learn

## :computer: Code

The code starts by importing the necessary libraries and loading the dataset. It then proceeds to clean the data by filling null values and converting non-integer values to integers.

Next, it splits the data into a training set and a validation set. Two machine learning models, RandomForestClassifier and LogisticRegression, are then created and fitted to the training data.

The models' performance is evaluated using accuracy, recall, and precision scores, and the best model is selected. Hyperparameter tuning is performed using GridSearchCV to optimize the model's performance.

Finally, the model is used to make predictions on the validation set, and these predictions are evaluated using the ROC or AUC curve.

## :chart_with_upwards_trend: Results

The results include accuracy, recall, and precision scores for both the training and validation sets. These scores are used to evaluate the model's performance and to select the best model.

## :file_folder: Files

- `train.csv`: The training dataset.
- `test.csv`: The test dataset.
- `smoker_prediction.ipynb`: The Jupyter notebook containing the code.

## :floppy_disk: Cloning the Repository

1. Navigate to the main page of the repository on GitHub.
2. Click on the green `Code` button above the list of files.
3. Copy the URL provided in the dropdown.
4. Open a Terminal (or Git Bash if you're on Windows).
5. Navigate to the directory where you want to clone the repository.
6. Type `git clone`, followed by the URL you copied earlier. It should look something like this:

`git clone https://github.com/pawaspy/Smoker-Status-Prediction`

7. Press `Enter` to create your local clone.

## :wrench: Installing Dependencies

1. Ensure that you have Python installed on your machine. If not, download it from [python.org](https://www.python.org/).
2. The project requires several Python libraries. These can be installed using `pip`, which comes installed with Python. Run the following command to install the necessary libraries:

`pip install pandas numpy matplotlib scikit-learn`


If there are any issues during installation, they might be due to incompatible versions of the libraries. Try installing them one by one in that case.

## :rocket: Running the Project

1. Navigate to the directory containing the project files in your terminal.
2. Run the Python script using the following command:

`python script_name.py`

Replace `script_name.py` with the name of the Python script you wish to run.

Note: This is a general guide. The actual process may vary depending on the specifics of the project. If the project has a `requirements.txt` file, you can install all required libraries at once using `pip install -r requirements.txt`. If the project is a Jupyter notebook, you can run it using Jupyter notebook or Jupyter lab.

## :bulb: Future Work

Future work could involve using more complex machine learning models and feature engineering to improve the model's performance. Additionally, the model could be tested on more diverse datasets to evaluate its generalizability.
