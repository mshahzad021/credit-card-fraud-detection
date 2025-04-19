import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# load data
df = pd.read_csv("credit_card_model_data.csv")

#features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prdict
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))

