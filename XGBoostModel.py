import os, joblib, xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score


class XGboost_Model:
    def __init__(self, DataSet, New_Accident_Data):
        self.Accident_Information = DataSet
        self.New_Accident_Information = New_Accident_Data
        self.Model_Path = os.path.join('Model', 'Model_Bundle.pkl')
        self.Results_Performance = None
        self.Predict = self.Predict_Accident_Severity()
    
    def Inside_Model_Folder(self):
        if os.path.exists(self.Model_Path):
            return joblib.load(self.Model_Path)
        else:
            return None

    def Save_Model(self, XGboost, Preprocessor_Bundle):
        Model_Bundle = {
            "Model": XGboost,
            "Preprocessor": Preprocessor_Bundle["Preprocessor"],
            "Label_Encoder": Preprocessor_Bundle["Label_Encoder"]
        }
        joblib.dump(Model_Bundle, self.Model_Path)
        
    def Preprocessor(self):
        Preprocessor = ColumnTransformer(
            transformers=[
                ("Missing", SimpleImputer(strategy="median"), ["Speed limit"]),
                ("Scaling", StandardScaler(), ["Number of Casualties", "Number of Vehicles", "Speed limit"]),
                ("Encoding", OneHotEncoder(dtype=float), ["Light Conditions", "Road Surface Conditions", 
                                                        "Road Type", "Weather Conditions"])                                  
            ]
        ) 
        Label_Encoder = LabelEncoder()
        
        return {"Preprocessor": Preprocessor, "Label_Encoder": Label_Encoder}
    
    def Model_Performance(self, XGboost, X_train, X_test, Y_train, Y_test, Label_Encoder):
        Y_pred_train = XGboost.predict(X_train)
        Y_pred_test = XGboost.predict(X_test)
        
        cm_train = confusion_matrix(Y_train, Y_pred_train)
        cm_test = confusion_matrix(Y_test, Y_pred_test)
        Accuracy_Train = accuracy_score(Y_train, Y_pred_train) * 100
        Accuracy_Test = accuracy_score(Y_test, Y_pred_test) * 100
        
        Original_Labels = Label_Encoder.inverse_transform(np.unique(Y_test))
        
        self.Results_Performance = {
            "Original_Labels": Original_Labels,
            "Accuracy Train": Accuracy_Train,
            "Accuracy Test": Accuracy_Test,
            "Confusion Matrix Train": cm_train,
            "Confusion Matrix Test": cm_test
        } 

    def XGboost_Model_Training(self):
        Loaded_Bundle = self.Inside_Model_Folder()
        Preprocess = self.Preprocessor()

        X = self.Accident_Information[
            [
                "Light Conditions", "Number of Casualties", "Number of Vehicles",       
                "Road Surface Conditions", "Road Type", "Speed limit", 
                "Urban or Rural Area", "Weather Conditions",
            ]
        ]
        Y = self.Accident_Information["Accident Severity"]
        
        X = Preprocess["Preprocessor"].fit_transform(X)
        Y = Preprocess["Label_Encoder"].fit_transform(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        if Loaded_Bundle is not None:
            Loaded_Model = Loaded_Bundle["Model"]
            Loaded_Model.fit(X_train, Y_train, xgb_model=Loaded_Model)
            XGboost = Loaded_Model
        else:
            XGboost = xgb.XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=100)
            XGboost.fit(X_train, Y_train)
            
        self.Save_Model(XGboost, Preprocess)
        self.Model_Performance(XGboost, X_train, X_test, Y_train, Y_test, Preprocess["Label_Encoder"])
        
    def Predict_Accident_Severity(self):
        if self.Inside_Model_Folder() is None:
            print("Model not found!! Training the model now...")
            self.XGboost_Model_Training()
            
        Loaded_Bundle = self.Inside_Model_Folder()
        Loaded_Model = Loaded_Bundle["Model"]
        Loaded_Preprocessor = Loaded_Bundle["Preprocessor"]
        Loaded_LabelEncoder = Loaded_Bundle["Label_Encoder"]

        New_Data_Processed = Loaded_Preprocessor.transform(self.New_Accident_Information)
        Accident_Severity = Loaded_Model.predict(New_Data_Processed)
        Accident_Severity_Label = Loaded_LabelEncoder.inverse_transform(Accident_Severity)

        return Accident_Severity_Label[0]