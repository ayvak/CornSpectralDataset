## Prediction of vomitoxin levels for corn samples from hyperspectral data
Analyzing and predicting the vomitoxin ppb value of a corn sample from hyperspectral data of corn

### Project Structure
CornSpectralDataset/   
│ ├── data/  
│ └── MLE-Assignment.csv # The dataset file   
│ ├── src/   
│ ├── processing_draft.py # Data preprocessing script   
│ ├── model_training.py # Model building and training script   
│ ├── evaluation.py # Model evaluation script   
│ └── main.py # Main script to run the project    
│ └── test_pipeline.py # Unit tests for the project   
│ ├── README.md # Project documentation   
│ └── requirements.txt # Python dependencies  

### Installation
   ```Clone the repository:
   git clone https://github.com/ayvak/CornSpectralDataset.git
   cd CornSpectralDataset
```

### Creating Virtual Environment with required libraries
```Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Running the Project
Data Preprocessing: The processing_draft.py script preprocesses the data by removing outliers, applying log transformations, and normalizing the features.

Model Training: The model_training.py script builds and trains the regression model.

Model Evaluation: The evaluation.py script evaluates the trained model using various metrics.

Main Script: The main.py script orchestrates the entire process from data preprocessing to model evaluation.

To run the main script:
python src/main.py

### Running the tests
```Test
pytest test_pipeline.py
```

### Explanation of different functions and files
1. src/processing_draft.py
This script handles data preprocessing, including:
a. Removing outliers using IQR.
b. Applying log transformations based on skewness.
c. Normalizing features using MinMaxScaler.

2. src/model_training.py
This script builds and trains the regression model using TensorFlow/Keras. It includes functions for:
a. Building the regression model.
b. Training the model with specified hyperparameters found using Optuna.

3. src/evaluation.py
This script evaluates the trained model using various metrics. It includes functions for:
a. Inverse transforming the predictions and target values.
b. Calculating evaluation metrics (MAE, RMSE, R2).
c. Visualizing by Plotting actual vs. predicted values.

4. src/main.py
This script orchestrates the entire process from data preprocessing to model evaluation. It includes:
a. Loading and preprocessing the data.
b. Applying Lasso regression for feature selection.
c. Training the neural network model.
d. Evaluating the model on training and test sets.

5. test_pipeline.py
This script contains unit tests for the data preprocessing, model building, and evaluation functions. It includes tests for:
a. process_data function
b. build_regression_model function
c. evaluate_regression function

### Acknowledgements
This project uses the Corn Spectral Dataset provided by imageio for their recruitment process.


