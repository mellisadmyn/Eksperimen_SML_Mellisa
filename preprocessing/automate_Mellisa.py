import os
import pickle
import shutil
import mlflow
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple

def categorize_blood_pressure(systolic: int, diastolic: int) -> str:
    """
    Categorizes blood pressure based on systolic and diastolic values.

    Parameters:
    systolic (int): The systolic blood pressure value.
    diastolic (int): The diastolic blood pressure value.

    Returns:
    str: The category of blood pressure

    Raises:
    None
    """
    if systolic < 90 or diastolic < 60:
        return "Low"
    elif 90 <= systolic < 120 and diastolic < 80:
        return "Normal"
    elif 120 <= systolic < 130 and diastolic < 80:
        return "Elevated"
    elif 130 <= systolic < 140 or 80 <= diastolic < 90:
        return "Hypertension Stage 1"
    elif 140 <= systolic < 180 or 90 <= diastolic < 120:
        return "Hypertension Stage 2"
    else:
        return "Hypertensive Crisis"

def preprocess_dataset(input_path: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocesses the raw sleep and lifestyle dataset including cleaning, feature engineering, encoding, scaling, until train-test split.

    Parameters:
    input_path (str): Path to the raw CSV dataset.
    output_dir (str): Path to save the preprocessed output files.

    Returns:
    tuple: A tuple containing cleaned dataframe, X_train, X_test, y_train, and y_test.

    Raises:
    FileNotFoundError        : [ERROR] Raised if the input file does not exist.
    ValueError               : [ERROR] Raised for data parsing or transformation issues.
    OSError                  : [ERROR] Raised if saving files fails (e.g., due to permission issues).
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[ERROR] Input file not found: {input_path}") from e
    except Exception as e:
        raise ValueError(f"[ERROR] Failed to read CSV file: {input_path}") from e

    try:
        # 5.1 Rename columns
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # 5.2 Drop irrelevant columns
        df.drop(columns=['person_id'], inplace=True)

        # 5.3 Handle missing value
        df['sleep_disorder'] = df['sleep_disorder'].fillna('No Issue')

        # 5.4 Handle incorrect value
        df['bmi_category'] = df['bmi_category'].replace('Normal Weight', 'Normal')

        # 5.5 Feature engineering: split and categorize blood pressure
        bp_split = df['blood_pressure'].str.split('/', expand=True)
        df = pd.concat([df, bp_split], axis=1).drop('blood_pressure', axis=1)
        df = df.rename(columns={0: 'blood_pressure_upper', 1: 'blood_pressure_lower'})
        df['blood_pressure_upper'] = df['blood_pressure_upper'].astype(int)
        df['blood_pressure_lower'] = df['blood_pressure_lower'].astype(int)
        df['blood_pressure_category'] = df.apply(
            lambda row: categorize_blood_pressure(row['blood_pressure_upper'], row['blood_pressure_lower']), axis=1
        )

        # 5.6 Drop duplicates
        df.drop_duplicates(inplace=True)

        # 5.7 Set categorical dtype
        df['gender'] = pd.Categorical(df['gender'], categories=['Male', 'Female'])
        df['occupation'] = pd.Categorical(df['occupation'], categories=[
            'Nurse', 'Doctor', 'Engineer', 'Lawyer', 'Teacher', 'Accountant',
            'Salesperson', 'Scientist', 'Software Engineer', 'Sales Representative', 'Manager'
        ])
        df['bmi_category'] = pd.Categorical(df['bmi_category'], categories=['Normal', 'Overweight', 'Obese'], ordered=True)
        df['blood_pressure_category'] = pd.Categorical(df['blood_pressure_category'], categories=[
            "Low", "Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2", "Hypertensive Crisis"
        ], ordered=True)
        df['sleep_disorder'] = pd.Categorical(df['sleep_disorder'], categories=['No Issue', 'Insomnia', 'Sleep Apnea'], ordered=True)

        # 5.8 Normalize numerical columns
        scaler = MinMaxScaler()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # 5.9 Encode categorical columns
        label_mappings = {}
        label_encoder = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['category']).columns

        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])
            label_mappings[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # 5.10 Train-test dataset split
        X = df.drop(columns=['sleep_disorder'])
        y = df['sleep_disorder']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Clean the output folder if it already exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save output files
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

        # Save label mappings
        mapping_path = os.path.join(output_dir, 'label_mappings.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(label_mappings, f)

        return df, X_train, X_test, y_train, y_test

    except Exception as e:
        raise ValueError(f"[ERROR] An error occurred during preprocessing: {str(e)}") from e


if __name__ == "__main__":
    input_file = "dataset_raw/Sleep_health_and_lifestyle_dataset.csv"
    output_dir = "preprocessing/outputs"

    try:
        mlflow.set_tracking_uri("file:./mlruns")

        with mlflow.start_run(run_name="Preprocessing_Run"):
            df_clean, X_train, X_test, y_train, y_test = preprocess_dataset(input_file, output_dir)

            mlflow.log_param("input_file", input_file)
            mlflow.log_param("output_dir", output_dir)
            mlflow.log_metric("rows_after_cleaning", df_clean.shape[0])

            mlflow.log_artifact(os.path.join(output_dir, 'X_train.csv'))
            mlflow.log_artifact(os.path.join(output_dir, 'X_test.csv'))
            mlflow.log_artifact(os.path.join(output_dir, 'y_train.csv'))
            mlflow.log_artifact(os.path.join(output_dir, 'y_test.csv'))
            mlflow.log_artifact(os.path.join(output_dir, 'label_mappings.pkl'))

    except Exception as err:
        print(f"[ERROR] Preprocessing failed: {err}")
