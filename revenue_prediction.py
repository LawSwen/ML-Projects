import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import  GridSearchCV,train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, make_scorer
import warnings
from sklearn.ensemble import GradientBoostingRegressor
import logging
import json
import zipfile
import pandas as pd
from copy import deepcopy
from copy import copy



warnings.filterwarnings('ignore')



def dataprocessing(train_path, test_path, target_col='revenue', max_features=20):
    dftrain = pd.read_json(train_path)
    dftest = pd.read_json(test_path)
    df_train = dftrain.drop(columns=["host", "name", "facilities", "listing_type"])   # Dealing with the traning data..
    df_test =  dftest.drop(columns=["host", "name", "facilities", "listing_type"])
    
    df_test.shape
    df_train.shape
    train_df = df_train.copy()
    test_df = df_test.copy()
    
    train_stats = {}  # Get the training stats to use on the test data as well.
    
    
    if target_col in train_df.columns:
        y_train = train_df[target_col]
        train_df = train_df.drop(columns=[target_col])
    else:
        y_train = None
    

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns   # Get the numeric and categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    
    # Handle missing values
    train_stats['numeric_medians'] = {}
    train_stats['categorical_modes'] = {}
    
    for col in numeric_cols:                       # Use median imputation for numeric features
        median_val = train_df[col].median()
        train_stats['numeric_medians'][col] = median_val
        train_df[col] = train_df[col].fillna(median_val)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(median_val)
    
    for col in categorical_cols:                 # Use the mode missing data treatment for the training data
        mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else 'Unknown'
        train_stats['categorical_modes'][col] = mode_val
        train_df[col] = train_df[col].fillna(mode_val)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(mode_val)      # Fill the same for the test data
    
    # 2. Conservative categorical encoding
    train_stats['label_encoders'] = {}        
    
    for col in categorical_cols:               # Use label_encoding for the traing and test data
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            train_stats['label_encoders'][col] = le
            
            if col in test_df.columns:
                # Safe transform for test data
                def safe_transform(x):
                    try:
                        return le.transform([str(x)])[0]
                    except ValueError:
                        return -1
                test_df[col] = test_df[col].astype(str).apply(safe_transform)
    
    # Basic Feature engineering focusing on skwed observations
    numeric_features = list(numeric_cols)
    
    for col in numeric_features[:3]:  # Limit to top 3
        if col in train_df.columns:
            # Check if feature is positive and skewed
            if train_df[col].min() >= 0 and train_df[col].skew() > 1:  # For the training data
                train_df[f'{col}_log'] = np.log1p(train_df[col])
                if col in test_df.columns:
                    test_df[f'{col}_log'] = np.log1p(test_df[col])      #  Conduct similiar transformation for the test data
    
                                                                         # Create Interaction feature
    if len(numeric_features) >= 2 and y_train is not None:
        correlations = train_df[numeric_features].corrwith(y_train).abs().sort_values(ascending=False)
        if len(correlations) >= 2:
            feat1, feat2 = correlations.index[0], correlations.index[1]
            train_df[f'{feat1}_x_{feat2}'] = train_df[feat1] * train_df[feat2]
            if feat1 in test_df.columns and feat2 in test_df.columns:
                test_df[f'{feat1}_x_{feat2}'] = test_df[feat1] * test_df[feat2]
    
    # 4. Remove extreme outliers for the target variable
    if y_train is not None:
        Q1, Q3 = y_train.quantile([0.1, 0.9])  # Use 10th and 90th percentiles
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (y_train >= lower_bound) & (y_train <= upper_bound)
        train_df = train_df[mask].reset_index(drop=True)  # Reset index after filtering
        y_train = y_train[mask].reset_index(drop=True)    # Reset index after filtering
    
                                                            # Select the maximum of 20 best features 
    if y_train is not None and len(train_df.columns) > max_features:
        selector = SelectKBest(score_func=f_regression, k=max_features)
        train_df_selected = pd.DataFrame(
            selector.fit_transform(train_df, y_train),
            columns=train_df.columns[selector.get_support()]
        ).reset_index(drop=True)  # Reset index
        
                                                             # Apply same selection to test data
        selected_features = train_df.columns[selector.get_support()]
        test_df_selected = test_df[selected_features].reset_index(drop=True)  # Reset index
        
        train_df, test_df = train_df_selected, test_df_selected
        train_stats['selected_features'] = selected_features
        
    return train_df, y_train, test_df  # Return the processed data 

train_df, y_train, test_df = dataprocessing("train.json", "test.json")
# print(train_df.shape, y_train.shape, test_df.shape)
print("\nData Processing Complete")
print("===========================")

print(f"\ntraining_df:{train_df.shape}, target_df{y_train.shape} Testing_df:{test_df.shape}" )
print("\n===Model Fitting===")
def model_fitting(train_df, y_train, test_df):
    X_tr, X_val, y_tr, y_val = train_test_split(                         #Splite the data
            train_df, y_train, test_size=0.4, random_state=42, stratify=None
        )
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_tr)
    X_val_test = scaler.transform(X_val)
    X_scaled_test = scaler.transform(test_df)
    mae_scorer = make_scorer(mean_absolute_error)
    param_grid = {
            'n_estimators': [1000],
            'learning_rate': [0.01],
            'max_depth': [200],
            'min_samples_split': [20],
            'min_samples_leaf': [10],
            'subsample': [1.0],
            'max_features': ['sqrt'],
            'loss': ['absolute_error']  # Optimized Paramters to use.. 
        }
        
    gbr = GradientBoostingRegressor(random_state=42, validation_fraction=0.1, n_iter_no_change=10)
    
    grid_search = GridSearchCV(
            estimator=gbr,
            param_grid=param_grid,
            cv=5,
            scoring=mae_scorer,
            n_jobs=-1,
            verbose=0,
            return_train_score=True, 
        )
    grid_search.fit(X_train_scaled, y_tr)
    best_gbr = grid_search.best_estimator_
    val_pred = best_gbr.predict(X_val_test)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"\nValidation MAE:{round(val_mae, 2)}")
   
    final_test = best_gbr.predict(X_scaled_test)
   
    print("\n===Model Fitting Finished===")
    
    print("===Prediction File Preparation===")
    predicted_df = pd.DataFrame({'revenue': final_test})
    predicted = predicted_df.to_dict(orient='records')
    with zipfile.ZipFile("model.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("predicted.json", json.dumps(predicted, indent=2))
    
    print("===Prediction Preparation File Done====")
    
    return val_mae

def solution():
    train_df, y_train, test_df = dataprocessing("train.json", "test.json")
    val_mea = model_fitting(train_df, y_train, test_df)
    
    return val_mea

if __name__ == '__main__':
    solution()
