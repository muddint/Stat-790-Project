import numpy as np
import pandas as pd
import gower
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import SingleTableMetadata



def calculate_privacy(real_data, synthetic_data):
    real_data_copy = real_data.copy()
    synthetic_data_copy = synthetic_data.copy()

    #convert all numerics to floats
    numeric_cols_original = real_data_copy.select_dtypes(include=['int64', 'float64']).columns
    real_data_copy[numeric_cols_original] = real_data_copy[numeric_cols_original].astype(float)
    numeric_cols_synthetic = synthetic_data_copy.select_dtypes(include=['int64', 'float64']).columns
    synthetic_data_copy[numeric_cols_synthetic] = synthetic_data_copy[numeric_cols_synthetic].astype(float)


    real_data_array = real_data_copy.to_numpy()
    synthetic_data_array = synthetic_data_copy.to_numpy()

    distances = gower.gower_matrix(real_data_array, synthetic_data_array)
    min_distances = np.min(distances, axis=1)
    gower_median = np.median(min_distances)
    privacy = 1 - gower_median
    return privacy

def calculate_utility(synthetic_data, test_data, target_column):
    X_train = synthetic_data.drop(target_column, axis=1)
    y_train = synthetic_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    # Encode categorical label
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        # Create and fit the encoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        # Combine categorical data for fitting
        combined_categorical_data = pd.concat([X_train[categorical_features], X_test[categorical_features]])
        encoder.fit(combined_categorical_data)

        # Transform the categorical features
        X_train_encoded = encoder.transform(X_train[categorical_features]).toarray()
        X_test_encoded = encoder.transform(X_test[categorical_features]).toarray()

        # Replace the categorical columns with the encoded data
        X_train = pd.concat([X_train.drop(categorical_features, axis=1), pd.DataFrame(X_train_encoded, index=X_train.index)], axis=1)
        X_test = pd.concat([X_test.drop(categorical_features, axis=1), pd.DataFrame(X_test_encoded, index=X_test.index)], axis=1)

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Initialize the RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on synthetic data
    clf.fit(X_train, y_train)

    # Predict on the real test data
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def evaluate(train_data, test_data, synthetic_data, label, visualization=False, directory=""):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)
    diagnostic_report = run_diagnostic(
        real_data=train_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        verbose=False
    ).get_properties()
    qr = evaluate_quality(
        real_data=train_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        verbose=False
    )
    quality_report = qr.get_properties()
    data_validity_score = diagnostic_report.loc[diagnostic_report['Property'] == 'Data Validity', 'Score'].values[0]
    data_structure_score = diagnostic_report.loc[diagnostic_report['Property'] == 'Data Structure', 'Score'].values[0]
    column_shapes_score = quality_report.loc[quality_report['Property'] == 'Column Shapes', 'Score'].values[0]
    column_pair_trends_score = quality_report.loc[quality_report['Property'] == 'Column Pair Trends', 'Score'].values[0]
    privacy = calculate_privacy(
        real_data=train_data,
        synthetic_data=synthetic_data
    )
    utility = calculate_utility(
        synthetic_data=synthetic_data,
        test_data=test_data,
        target_column=label
    )

    results = {
        "data_validity_score": data_validity_score,
        "data_structure_score": data_structure_score,
        "column_fidelity": column_shapes_score,
        "row_fidelity": column_pair_trends_score,
        "privacy_score": privacy,
        "utility_score": utility
    }
    
    if visualization:
        fig = qr.get_visualization(property_name='Column Shapes')
        fig.write_image(directory + '/column-shapes.png')
        fig = qr.get_visualization(property_name='Column Pair Trends')
        fig.write_image(directory + '/column-pairs.png')

    print("Data Validity Score:", data_validity_score)
    print("Data Structure Score:", data_structure_score)
    print("Column Fidelity:", column_shapes_score)
    print("Row Fidelity:", column_pair_trends_score)
    print("Privacy Score:", privacy)
    print("Utility Score:", utility)
    
    return results

