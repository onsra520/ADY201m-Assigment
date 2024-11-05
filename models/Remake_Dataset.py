import os, shutil
import pandas as pd
        
def Columns_Remake():
    Dataset_Remake = pd.read_csv(os.path.join("UK Accident Dataset", "Accident_Information.csv"), encoding="latin1", low_memory=False).copy()
    Dataset_Remake.columns = [col.replace("_", " ") for col in Dataset_Remake.columns]
    if "Date" in Dataset_Remake.columns:
        Dataset_Remake["Date"] = (
            pd.to_datetime(Dataset_Remake["Date"], format="%Y-%m-%d", errors="coerce")
        ).dt.strftime("%d-%m-%Y")
    elif "Time" in Dataset_Remake.columns:
        Dataset_Remake["Time"] = (
            pd.to_datetime(Dataset_Remake["Time"], format="%H:%M", errors="coerce")
        ).dt.strftime("%H:%M:%S")
        
    Dataset_Remake.to_csv(
        os.path.join("UK Accident Dataset", "Accident_Information_Remake.csv"),
        index=False,
    )        

def Remake():
    os.makedirs("UK Accident Dataset", exist_ok=True)
    os.makedirs("Model", exist_ok=True)     
    if "Accident_Information.csv" in os.listdir():
        shutil.move("Accident_Information.csv", "UK Accident Dataset")
        Columns_Remake()
    return pd.read_csv(os.path.join("UK Accident Dataset", "Accident_Information_Remake.csv"), encoding="latin1", low_memory=False).copy()