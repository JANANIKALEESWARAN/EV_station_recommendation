import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from pytorch_tabnet.tab_model import TabNetClassifier

# --------------------
# Load Dataset
# --------------------

df = pd.read_csv("data/train.csv")

# --------------------
# Encode Categorical Columns
# --------------------

vehicle_encoder = LabelEncoder()
time_encoder = LabelEncoder()

df["Vehicle"] = vehicle_encoder.fit_transform(df["Vehicle"])
df["Time"] = time_encoder.fit_transform(df["Time"])

# --------------------
# Features
# --------------------

X = df.drop("Selected",axis=1)

y = df["Selected"]

# --------------------
# Split Dataset
# --------------------

X_train,X_test,y_train,y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# --------------------
# Convert to numpy
# --------------------

X_train = X_train.values
X_test = X_test.values

y_train = y_train.values
y_test = y_test.values

# --------------------
# Train Model
# --------------------

model = TabNetClassifier()

model.fit(

    X_train,

    y_train,

    eval_set=[(X_test,y_test)],

    max_epochs=5,

    patience=10,

    batch_size=1024,

    virtual_batch_size=128

)

# --------------------
# Prediction
# --------------------

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print()

print("Accuracy :",accuracy)

# --------------------
# Save Model
# --------------------

model.save_model("models/tabnet_model")