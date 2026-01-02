# House-price-prediction-model
This project predicts residential house prices using the Ames Housing Dataset. The goal is to build and compare multiple regression models on tabular real-estate data, while practicing data cleaning, feature engineering, and evaluation of ML models.

The notebook walks through:
data cleaning & preprocessing,
encoding categorical variables,
feature engineering,
training multiple models,
evaluating and comparing results.


# Dataset
The project uses: train.csv (standard Kaggle House Prices dataset)
Each row represents a house and its properties such as:

lot size
basement quality
garage attributes
number of rooms
year built
neighborhood
exterior materials
Target variable:
SalePrice
ID column removed during preprocessing.



# Data Cleaning & Preprocessing
Key steps performed:
dropped high-missing / low-value columns
(Alley, PoolQC, Fence, MiscFeature)
filled categorical NaNs with "None"
filled numerical NaNs with 0 or median values
handled basement & garage missing values carefully
imputed Electrical with mode
applied one-hot encoding to all object-type features
removed Id from feature set
Encoding performed via:
cat_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)



# Train / Test Split
from sklearn.model_selection import train_test_split
y = df["SalePrice"]
X = df.drop(columns=["SalePrice", "Id"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Models Trained
Baseline & ML ensemble models:
Linear Regression
Random Forest Regressor
XGBoost Regressor
Evaluation metrics:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score



# Model Comparison
Each model was evaluated on test data and compared using:
MAE → average prediction error
RMSE → penalizes larger errors
R² → variance explained by model
XGBoost and Random Forest generally outperform the linear baseline on this dataset.
(Results may vary depending on parameters and data splits.)



# Tech Stack
Python
pandas
NumPy
scikit-learn
XGBoost
