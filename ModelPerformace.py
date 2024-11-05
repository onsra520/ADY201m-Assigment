import seaborn as sns
import matplotlib.pyplot as plt


def Confusion_Matrix_Train(Model_Results):
    print(f'Accuracy for Train set: {Model_Results["Accuracy Train"]}%')
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="white")
    sns.heatmap(
        Model_Results["Confusion Matrix Train"],
        annot=True,
        fmt="d",
        cmap="PuRd",
        xticklabels=Model_Results["Original_Labels"],
        yticklabels=Model_Results["Original_Labels"],
    )
    plt.title("Confusion Matrix for Train Set (XGBoost)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def Confusion_Matrix_Test(Model_Results):
    print(f'Accuracy for Test set: {Model_Results["Accuracy Test"]}%')    
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="white")
    sns.heatmap(
        Model_Results["Confusion Matrix Test"],
        annot=True,
        fmt="d",
        cmap="PuRd",
        xticklabels=Model_Results["Original_Labels"],
        yticklabels=Model_Results["Original_Labels"],
    )
    plt.title("Confusion Matrix for Test Set (XGBoost)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
