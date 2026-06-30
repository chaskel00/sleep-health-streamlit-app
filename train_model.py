import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report


# load the data
df = pd.read_csv("data/sleep_health_and_lifestyle_dataset.csv")

# blank sleep disorder means no disorder
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

# split blood pressure into two columns
df[["Systolic", "Diastolic"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
df = df.drop(columns=["Blood Pressure"])

# ordinal encoding for BMI
bmi_map = {
    "Normal": 0,
    "Normal Weight": 0,
    "Overweight": 1,
    "Obese": 2
}

df["BMI Category"] = df["BMI Category"].map(bmi_map)

# one hot encode occupation and gender
df = pd.get_dummies(df, columns=["Occupation", "Gender"], drop_first=True)

# set up x and y
X = df.drop(["Sleep Disorder", "Person ID"], axis=1)
y = df["Sleep Disorder"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# random forest settings from the original project
rf_base = RandomForestClassifier(
    n_estimators=50,
    max_depth=7,
    random_state=42
)

# feature selection
rfe = RFE(estimator=rf_base, n_features_to_select=4)
rfe.fit(X_train_scaled, y_train)

X_train_final = rfe.transform(X_train_scaled)
X_test_final = rfe.transform(X_test_scaled)

# final model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=7,
    random_state=42
)

model.fit(X_train_final, y_train)

# test model
predictions = model.predict(X_test_final)
accuracy = accuracy_score(y_test, predictions)

print("Random Forest Model Results")
print("---------------------------")
print("Accuracy:", round(accuracy, 4))
print()
print(classification_report(y_test, predictions))

print("Selected features:")
print(list(X.columns[rfe.support_]))

# save everything the app needs
model_info = {
    "model": model,
    "scaler": scaler,
    "rfe": rfe,
    "columns": X.columns.tolist(),
    "selected_features": list(X.columns[rfe.support_])
}

joblib.dump(model_info, "sleep_model.pkl")

print()
print("Model saved as sleep_model.pkl")