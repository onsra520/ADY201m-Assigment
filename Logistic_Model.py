import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class Logistic_Model:
    def __init__(self, data, new_data=None, model_path="Model\\logistic_model.pkl"):
        self.data = data
        self.new_data = new_data
        self.model_path = model_path
        self.model = LogisticRegression(max_iter=1000)
        self.smote = SMOTE(random_state=42)
        self.preprocessor = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_pred_test = None
        self.y_pred_train = None

        self.Results_Performance = {"Original_Labels": None, 
                                    "Accuracy Train": None, 
                                    "Accuracy Test": None, 
                                    "Confusion Matrix Train": None, 
                                    "Confusion Matrix Test": None 
                                    }
        #Check xem model sẵn chưa
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
            print("Model Loaded.")
        else:
            self.preprocess_data()
            self.train_model()
            self.evaluate_model()
            self.save_model(self.model_path)
            print("Model trained and saved.")

        self.Predict = self.predict_new_data()
    def preprocess_data(self):
        self.X = self.data[['Light Conditions', 'Number of Casualties', 'Number of Vehicles', 'Road Surface Conditions',
                            'Road Type', 'Speed limit', 'Weather Conditions']]
        self.y = self.data['Accident Severity'] 

        numeric_features = ['Number of Casualties', 'Speed limit', 'Number of Vehicles']
        categorical_features = ['Light Conditions', 'Road Surface Conditions', 'Road Type', 'Weather Conditions']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.X = self.preprocessor.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)

    def train_model(self):
        X_res, y_res = self.smote.fit_resample(self.X_train, self.y_train)
        self.model.fit(X_res, y_res)
        self.Results_Performance["Accuracy Train"] = self.model.score(X_res, y_res)*100

    def evaluate_model(self):
        self.y_pred_test = self.model.predict(self.X_test)
        self.y_pred_train = self.model.predict(self.X_train)
        self.Results_Performance["Original_Labels"] = sorted(self.y_test.unique()) 
        self.Results_Performance["Accuracy Test"] = self.model.score(self.X_test, self.y_test)*100 
        self.Results_Performance["Confusion Matrix Train"] = confusion_matrix(self.y_train, self.y_pred_train) 
        self.Results_Performance["Confusion Matrix Test"] = confusion_matrix(self.y_test, self.y_pred_test)

        self.classification_rep = classification_report(self.y_test, self.y_pred_test, output_dict=True)
        self.confusion_mat = confusion_matrix(self.y_test, self.y_pred_test)
        print(pd.DataFrame(self.classification_rep).transpose())
        print(self.confusion_mat)

    
    def save_model(self, filename): 
        os.makedirs(os.path.dirname(filename), exist_ok=True) 
        joblib.dump((self.model, self.preprocessor, self.Results_Performance), filename) 
    
    def load_model(self, filename): 
        self.model, self.preprocessor, self.Results_Performance = joblib.load(filename)
    
    def predict_new_data(self):
        if self.new_data is not None:
            new_data_processed = self.preprocessor.transform(self.new_data)
            prediction = self.model.predict(new_data_processed)
            return prediction
        else:
            raise ValueError("No new data for prediction.")
