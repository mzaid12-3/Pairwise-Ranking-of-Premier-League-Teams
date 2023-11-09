import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, numerical_cols, categorical_cols):
    df = df.copy()

    # Fill missing numerical values with mean
    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    # Standardize numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Fill missing categorical values with mode
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=categorical_cols)

    return df