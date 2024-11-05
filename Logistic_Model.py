import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class Logistic_Model:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.model = LogisticRegression(max_iter=1000)
        self.smote = SMOTE(random_state=42)
        self.preprocessor = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_pred = None
        self.Results_Performance = { "Original_Labels": None, 
                                    "Accuracy Train": None, 
                                    "Accuracy Test": None, 
                                    "Confusion Matrix Train": None, 
                                    "Confusion Matrix Test": None 
                                    }

    
    def preprocess_data(self):
        self.X = self.data[['Light Conditions', 'Number of Casualties', 'Number of Vehicles', 'Road Surface Conditions',
                            'Road Type', 'Speed limit', 'Weather Conditions']]
        self.y = self.data['Accident Severity'] 

        numeric_features = ['Number of Casualties', 'Speed limit', 'Number of Vehicles']
        categorical_features = ['Light Conditions', 'Road Surface Conditions', 'Road Type', 'Weather Conditions']

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

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
        self.Results_Performance["Accuracy Train"] = self.model.score(X_res, y_res)

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test)
        self.Results_Performance["Original_Labels"] = self.y_test 
        self.Results_Performance["Accuracy Test"] = self.model.score(self.X_test, self.y_test) 
        self.Results_Performance["Confusion Matrix Train"] = confusion_matrix(self.y_train, self.model.predict(self.X_train)) 
        self.Results_Performance["Confusion Matrix Test"] = confusion_matrix(self.y_test, self.y_pred)

        self.classification_rep = classification_report(self.y_test, self.y_pred, output_dict=True)
        self.confusion_mat = confusion_matrix(self.y_test, self.y_pred)
        print(pd.DataFrame(self.classification_rep).transpose())
        print(self.confusion_mat)

    
    def save_model(self, filename):
        joblib.dump(self.model, filename)
    
    def load_model(self, filename):
        self.model = joblib.load(filename)
    
    def predict_new_data(self, new_data):
        new_data_processed = self.preprocessor.transform(new_data)
        prediction = self.model.predict(new_data_processed)
        return prediction