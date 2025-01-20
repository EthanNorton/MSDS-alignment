import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.nonparametric.smoothers_lowess import lowess

# Additional imports for new functionality
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.feature_selection import SelectFromModel, RFE
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data
house_prices_train_data = pd.read_csv('/content/drive/MyDrive/MSDS422 - Group 1/ Module 1: Machine Learning Framework/house-prices-advanced-regression-techniques/train.csv')
house_prices_test_data = pd.read_csv('/content/drive/MyDrive/MSDS422 - Group 1/ Module 1: Machine Learning Framework/house-prices-advanced-regression-techniques/test.csv')

# Prepare the data
X_df = house_prices_train_data.drop(['SalePrice', 'Id'], axis=1)
y = house_prices_train_data['SalePrice']


def engineer_features(df):
    """Engineer all features based on the notebook"""
    df = df.copy()

    # Create dictionary to store highest value categories
    highest_categories = {
        'MSZoning': 'FV',          # 214014
        'Street': 'Pave',          # 181130
        'Alley': 'Pave',          # 168000
        'LotShape': 'IR2',         # 239833
        'LandContour': 'HLS',      # 231533
        'Utilities': 'AllPub',     # 180950
        'LotConfig': 'CulDSac',    # 223854
        'LandSlope': 'Sev',        # 204379
        'Neighborhood': 'NoRidge',  # 335295
        'Condition1': 'PosA',      # 225875
        'Condition2': 'PosA',      # 325000
        'BldgType': '1Fam',        # 185763
        'HouseStyle': '2.5Fin',    # 220000
        'RoofStyle': 'Shed',       # 225000
        'RoofMatl': 'WdShngl',     # 390250
        'Exterior1st': 'ImStucc',  # 262000
        'Exterior2nd': 'Other',    # 319000
        'MasVnrType': 'Stone',     # 265583
        'ExterQual': 'Ex',         # 367360
        'ExterCond': 'Ex',         # 201333
        'Foundation': 'PConc',      # 225230
        'BsmtQual': 'Ex',          # 327041
        'BsmtCond': 'Gd',          # 213599
        'BsmtExposure': 'Gd',      # 257689
        'BsmtFinType1': 'GLQ',     # 235413
        'BsmtFinType2': 'ALQ',     # 209942
        'Heating': 'GasA',         # 182021
        'HeatingQC': 'Ex',         # 214914
        'CentralAir': 'Y',         # 186186
        'Electrical': 'SBrkr',     # 186825
        'KitchenQual': 'Ex',       # 328554
        'Functional': 'Typ',       # 183429
        'FireplaceQu': 'Ex',       # 337712
        'GarageType': 'BuiltIn',   # 254751
        'GarageFinish': 'Fin',     # 240052
        'GarageQual': 'Ex',        # 241000
        'GarageCond': 'TA',        # 187885
        'PavedDrive': 'Y',         # 186433
        'PoolQC': 'Ex',            # 490000
        'Fence': 'GdPrv',          # 178927
        'MiscFeature': 'TenC',     # 250000
        'SaleType': 'New',         # 274945
        'SaleCondition': 'Partial' # 272291
    }

    # Create dummy variables for highest value categories
    for col, highest_cat in highest_categories.items():
        if col in df.columns:  # Check if column exists in dataframe
            dummy_name = f'{col}_Premium'
            df[dummy_name] = (df[col] == highest_cat).astype(int)
      # Add new significant interactions
    df['GrLivArea_OverallQual'] = df['GrLivArea'] * df['OverallQual']
    df['OverallQual_GarageArea'] = df['OverallQual'] * df['GarageArea']
    df['OverallQual_TotalBsmtSF'] = df['OverallQual'] * df['TotalBsmtSF']
    df['GrLivArea_YearBuilt'] = df['GrLivArea'] * df['YearBuilt']
    df['OverallQual_YearBuilt'] = df['OverallQual'] * df['YearBuilt']
    # Total Square Footage
    df['TotalSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF'] + \
                    df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + \
                    df['LowQualFinSF'] + df['GrLivArea']
    # House Age and Remodeling Features
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['HouseAgeLessThan25'] = (df['HouseAge'] < 100).astype(int)

    # Age Effect
    df['AgeEffect'] = df['HouseAge'].apply(lambda x:
        -1551 * x + 235586 if x < 100 else 887 * x + 52089)
    df['AgeEffect_Normalized'] = (df['AgeEffect'] - df['AgeEffect'].mean()) / df['AgeEffect'].std()

    # Binary Features
    df['Has_Garage'] = (df['GarageType'].notna()).astype(int)
    df['Has_Basement'] = (df['BsmtQual'].notna()).astype(int)
    df['Is_Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['Has_Pool'] = (df['PoolArea'] > 0).astype(int)
    df['Has_Fireplace'] = (df['Fireplaces'] > 0).astype(int)

    # Premium House Features
    df['IsHighQual'] = (df['OverallQual'] > 6).astype(int)
    df['HasSF'] = (df['TotalSF'] > 0).astype(int)
    df['PremiumHouse'] = df['TotalSF'] * ((df['IsHighQual'] == 1) & (df['HasSF'] == 1))

    # Polynomial Features
    df['TotalSF_Squared'] = df['TotalSF'] ** 2
    df['TotalSF_Cubed'] = df['TotalSF'] ** 3
    df['OverallQual_Squared'] = df['OverallQual'] ** 2
    df['OverallQual_Cubed'] = df['OverallQual'] ** 3
    df['YearBuilt_Squared'] = df['YearBuilt'] ** 2
    df['YearBuilt_Cubed'] = df['YearBuilt'] ** 3

    # Quality-Size Interactions
    df['Quality_Size_Interaction'] = df['OverallQual'] * df['TotalSF']
    df['Quality_Size_Squared'] = df['OverallQual'] * df['TotalSF_Squared']

    # Add new significant interactions
    df['GrLivArea_OverallQual'] = df['GrLivArea'] * df['OverallQual']
    df['OverallQual_GarageArea'] = df['OverallQual'] * df['GarageArea']
    df['OverallQual_TotalBsmtSF'] = df['OverallQual'] * df['TotalBsmtSF']
    # Handle room counts (combine or simplify)
    df['RoomRatio'] = df['TotRmsAbvGrd'] / df['BedroomAbvGr']

    # Simplify garage features
    df['HasGarage'] = (df['GarageCars'] > 0).astype(int)
    df['GarageScore'] = df['GarageCars'] * df['GarageArea'] / 1000  # Normalized interaction

    # Create simplified quality score
    df['QualityScore'] = (df['OverallQual'] + df['OverallCond']) / 2

    columns_to_drop = [
        # Square footage columns
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',

        # Year-related
        'GarageYrBlt', 'YearRemodAdd', 'YearBuilt', 'YrSold',

        # Room-related
        'TotRmsAbvGrd', 'BedroomAbvGr', 'KitchenAbvGr',

        # Garage-related
        'GarageCars', 'GarageArea',

        # Quality-related
        'OverallQual_YearBuilt', 'OverallCond'
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df
def auto_select_transformation(X, y, feature, spearman_threshold=0.6, unique_ratio_threshold=0.05, max_threshold=10):
    """
    Automatically select the best transformation for a feature based on:
    1. Initial screening criteria
    2. Performance metrics for each transformation
    3. AIC/BIC to penalize complexity
    """
    from scipy import stats
    from sklearn.preprocessing import PowerTransformer
    from sklearn.metrics import r2_score
    import numpy as np

    results = {}

    # Initial screening
    unique_ratio = len(X[feature].unique()) / len(X[feature])
    max_value = X[feature].max()
    original_spearman = abs(stats.spearmanr(X[feature], y)[0])

    # Skip if feature doesn't meet basic criteria
    if not (original_spearman > spearman_threshold and
            unique_ratio > unique_ratio_threshold and
            max_value > max_threshold):
        return None, 'no_transform', {}

    # Dictionary of transformation functions
    transformations = {
        'original': lambda x: x,
        'log': lambda x: np.log1p(x) if (x > 0).all() else None,
        'sqrt': lambda x: np.sqrt(x) if (x >= 0).all() else None,
        'box-cox': lambda x: stats.boxcox(x)[0] if (x > 0).all() else None,
        'yeo-johnson': lambda x: PowerTransformer(method='yeo-johnson')
                                .fit_transform(x.values.reshape(-1, 1)).ravel(),
        'polynomial': lambda x: np.column_stack([x, x**2])
    }

    # Evaluate each transformation
    for name, transform_func in transformations.items():
        try:
            # Apply transformation
            if name == 'polynomial':
                transformed = transform_func(X[feature])
                # Use first polynomial term for correlation
                transformed_feature = transformed[:, 0]
            else:
                transformed_feature = transform_func(X[feature])

            if transformed_feature is None:
                continue

            # Calculate metrics
            spearman = abs(stats.spearmanr(transformed_feature, y)[0])
            pearson = abs(stats.pearsonr(transformed_feature, y)[0])
            r2 = r2_score(y, np.poly1d(np.polyfit(transformed_feature, y, 1))(transformed_feature))

            # Calculate AIC (Akaike Information Criterion)
            n = len(y)
            k = 2 if name == 'polynomial' else 1  # Number of parameters
            residuals = y - np.poly1d(np.polyfit(transformed_feature, y, 1))(transformed_feature)
            aic = n * np.log(np.sum(residuals**2)/n) + 2*k

            results[name] = {
                'spearman': spearman,
                'pearson': pearson,
                'r2': r2,
                'aic': aic,
                'complexity': k
            }

        except Exception as e:
            print(f"Error with {name} transformation for {feature}: {str(e)}")
            continue

    if not results:
        return None, 'no_transform', {}

    # Select best transformation based on composite score
    # Weight different metrics
    for name in results:
        results[name]['composite_score'] = (
            0.4 * results[name]['r2'] +
            0.3 * results[name]['spearman'] +
            0.3 * results[name]['pearson'] -
            0.1 * results[name]['complexity']  # Penalty for complexity
        )

    # Find best transformation
    best_transform = max(results.items(), key=lambda x: x[1]['composite_score'])[0]

    # Only transform if it's significantly better than original
    if (best_transform != 'original' and
        results[best_transform]['composite_score'] > results['original']['composite_score'] * 1.05):
        return transformations[best_transform](X[feature]), best_transform, results
    else:
        return X[feature], 'original', results

def auto_transform_features(X, y):
    """Apply automatic transformation to all eligible features"""
    X_transformed = X.copy()
    transformation_summary = {}

    for feature in X.select_dtypes(include=['int64', 'float64']).columns:
        transformed_feature, method, metrics = auto_select_transformation(X, y, feature)

        if method != 'no_transform':
            if method != 'original':
                X_transformed[f'{feature}_{method}'] = transformed_feature
                transformation_summary[feature] = {
                    'method': method,
                    'metrics': metrics
                }

    # Print summary
    print("\nTransformation Summary:")
    for feature, info in transformation_summary.items():
        if info['method'] != 'original':
            print(f"\n{feature}:")
            print(f"  Best transformation: {info['method']}")
            print(f"  Improvement in R²: {info['metrics'][info['method']]['r2'] - info['metrics']['original']['r2']:.4f}")

    return X_transformed, transformation_summary

# Use the function
X_transformed, summary = auto_transform_features(X_df, y)

# Apply to dataframe
def transform_appropriate_features(X_df, y):
    """Apply log transformations only to appropriate features"""
    X_transformed = X_df.copy()
    transformed_features = []

    for col in X_df.select_dtypes(include=['int64', 'float64']).columns:
        if should_transform_feature(X_df, col, y):
            X_transformed[f'{col}_log'] = np.log1p(X_df[col])
            transformed_features.append(col)

    print("\nSummary:")
    print(f"Total features transformed: {len(transformed_features)}")
    print("\nTransformed features:")
    for feature in transformed_features:
        print(f"- {feature}")

    return X_transformed, transformed_features

    # Convert to DataFrame
    results_df = pd.DataFrame(results).sort_values('original_spearman', ascending=False)

    return df_transformed, results_df

# Apply the transformation
X_df_transformed, transformation_results = identify_and_transform_spearman_features(X_df, y)

# Display summary of transformations
print("\nSummary of Transformations:")
print(transformation_results)
def select_best_features(X, y, n_features=20):
    """
    Perform feature selection using multiple methods
    """
    feature_methods = {}
    
    # Lasso Selection
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X, y)
    feature_methods['lasso'] = pd.Series(
        np.abs(lasso.coef_), 
        index=X.columns
    ).sort_values(ascending=False)
    
    # Random Forest Importance
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    feature_methods['rf'] = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    
    # Recursive Feature Elimination
    rfe = RFE(
    estimator=RandomForestRegressor(random_state=42),
    n_features_to_select=min(20, X.shape[1]//2)  # Limit feature selection
)
    rfe.fit(X, y)
    feature_methods['rfe'] = pd.Series(
        rfe.ranking_,
        index=X.columns
    ).sort_values()
    
    return feature_methods
    # New function for creating ensemble models
def create_ensemble_models():
    """
    Create various ensemble models
    """
    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgb', LGBMRegressor(random_state=42))
    ]
    
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
        'Voting': VotingRegressor(estimators=base_models),
        'Stacking': StackingRegressor(
            estimators=base_models,
            final_estimator=GradientBoostingRegressor(random_state=42)
        )
    }
    
    return models
def create_shrinkage_models(alphas=np.logspace(-4, 4, 100)):
    """
    Create models with explicit shrinkage penalties
    Returns dictionary of models with different regularization approaches
    """
    shrinkage_models = {
        # L2 regularization (Ridge)
        'Ridge': RidgeCV(
            alphas=alphas,
            cv=5,
            scoring='neg_mean_squared_error'
        ),
        
        # L1 regularization (Lasso)
        'Lasso': LassoCV(
            alphas=alphas,
            cv=5,
            max_iter=2000,
            selection='random',
            random_state=42
        ),
        
        # L1 + L2 regularization (ElasticNet)
        'ElasticNet': ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            alphas=alphas,
            cv=5,
            max_iter=2000,
            random_state=42
        ),
        
        # Robust regression with adaptive shrinkage
        'HuberRegressor': HuberRegressor(
            epsilon=1.35,
            alpha=0.0001,
            max_iter=2000
        )
    }
    
    return shrinkage_models

def analyze_shrinkage_effects(X, y, models):
    """
    Analyze and visualize effects of different shrinkage penalties
    """
    results = {}
    coefficients = {}
    
    # Fit models and collect coefficients
    for name, model in models.items():
        model.fit(X, y)
        
        if hasattr(model, 'alpha_'):
            best_alpha = model.alpha_
        elif hasattr(model, 'best_params_'):
            best_alpha = model.best_params_.get('alpha', None)
        else:
            best_alpha = None
            
        if hasattr(model, 'coef_'):
            coefficients[name] = model.coef_
        
        results[name] = {
            'best_alpha': best_alpha,
            'n_nonzero_coefs': np.sum(np.abs(coefficients[name]) > 1e-6)
        }
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot coefficient paths
    plt.subplot(2, 2, 1)
    for name, coef in coefficients.items():
        plt.plot(range(len(coef)), np.sort(np.abs(coef))[::-1], 
                label=f'{name} (nonzero: {results[name]["n_nonzero_coefs"]})')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Absolute Coefficient Value')
    plt.title('Coefficient Paths')
    plt.legend()
    plt.yscale('log')
    
    # Plot number of non-zero coefficients
    plt.subplot(2, 2, 2)
    nonzero_coefs = [results[name]['n_nonzero_coefs'] for name in results.keys()]
    plt.bar(results.keys(), nonzero_coefs)
    plt.title('Number of Non-zero Coefficients')
    plt.xticks(rotation=45)
    
    # If using Ridge, plot Ridge trace
    if 'Ridge' in models:
        ridge_model = models['Ridge']
        if hasattr(ridge_model, 'alphas_') and hasattr(ridge_model, 'coef_path_'):
            plt.subplot(2, 2, 3)
            for coef_path in ridge_model.coef_path_.T:
                plt.plot(ridge_model.alphas_, coef_path)
            plt.xscale('log')
            plt.xlabel('Alpha (regularization strength)')
            plt.ylabel('Coefficient value')
            plt.title('Ridge Trace')
    
    plt.tight_layout()
    plt.show()
    
    return results


def analyze_feature_types(X_df):
    """Identify binary and continuous features"""
    binary_features = []
    continuous_features = []

    for column in X_df.columns:
        if len(X_df[column].unique()) == 2:
            binary_features.append(column)
        else:
            continuous_features.append(column)

    return binary_features, continuous_features

def analyze_feature_importance(model, feature_names, binary_features, X_original):
    """Analyze feature importance considering binary vs continuous"""
    if hasattr(model, 'best_estimator_'):
        coef = model.best_estimator_.coef_
    else:
        coef = model.coef_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Type': ['Binary' if f in binary_features else 'Continuous' for f in feature_names]
    })

    importance_df['Std_Impact'] = importance_df.apply(lambda x:
        abs(x['Coefficient']) if x['Type'] == 'Binary'
        else abs(x['Coefficient']) * X_original[x['Feature']].std(),
        axis=1
    )

    return importance_df.sort_values('Std_Impact', ascending=False)

def compare_reduction_methods(X_scaled, y_log, n_components=0.95):
    """
    Compare PCA, feature selection, and combined approaches
    Returns dictionary of transformed datasets
    """
    reduced_datasets = {}
    
    # 1. PCA - simplified
    pca = PCA(n_components=min(10, X_scaled.shape[1]))  # Limit to max 10 components
    X_pca = pca.fit_transform(X_scaled)
    
    # 2. Feature Selection - using simpler method
    # Use correlation instead of complex models
    correlations = pd.DataFrame({
        'feature': X_scaled.columns,
        'correlation': [abs(np.corrcoef(X_scaled[col], y_log)[0,1]) for col in X_scaled.columns]
    }).sort_values('correlation', ascending=False)
    
    # Select top 10 features
    top_features = correlations['feature'].head(10).tolist()
    X_selected = X_scaled[top_features]
    
    # 3. Combined Approach (PCA on selected features)
    X_combined = pca.fit_transform(X_selected)
    
    reduced_datasets = {
        'original': X_scaled,
        'pca': X_pca,
        'feature_selection': X_selected,
        'combined': X_combined
    }
    
    # Simple visualization
    plt.figure(figsize=(10, 4))
    
    # Plot explained variance ratio
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Components')
    plt.ylabel('Cumulative Variance Ratio')
    plt.title('PCA Variance')
    
    # Plot top correlations
    plt.subplot(1, 2, 2)
    plt.bar(range(10), correlations['correlation'].head(10))
    plt.xticks(range(10), correlations['feature'].head(10), rotation=45, ha='right')
    plt.title('Top 10 Feature Correlations')
    
    plt.tight_layout()
    plt.show()
    
    return reduced_datasets
def identify_and_transform_spearman_features(X_df, y, spearman_threshold=0.1):
    """Identify features with strong Spearman correlation and apply transformations"""
    from scipy import stats

    df_transformed = X_df.copy()
    results = []

    # Analyze each numeric feature
    for column in X_df.select_dtypes(include=['int64', 'float64']).columns:
        # Calculate original Spearman correlation
        original_spearman = abs(stats.spearmanr(X_df[column], y, nan_policy='omit')[0])

        if pd.isna(original_spearman):
            continue

        # Only transform features with meaningful correlation
        if original_spearman > spearman_threshold:
            # Try log transformation for positive values
            if (X_df[column] > 0).all():
                try:
                    log_transformed = np.log1p(X_df[column])
                    log_spearman = abs(stats.spearmanr(log_transformed, y)[0])

                    # Add log transform if it improves correlation
                    if log_spearman > original_spearman:
                        df_transformed[f'{column}_log'] = log_transformed
                        transform_used = 'log'
                        final_spearman = log_spearman
                    else:
                        transform_used = 'none'
                        final_spearman = original_spearman
                except Exception as e:
                    print(f"Error transforming {column}: {str(e)}")
                    transform_used = 'error'
                    final_spearman = original_spearman
            else:
                transform_used = 'skipped'
                final_spearman = original_spearman

            # Store results
            results.append({
                'feature': column,
                'original_spearman': original_spearman,
                'final_spearman': final_spearman,
                'transformation': transform_used
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).sort_values('original_spearman', ascending=False)

    return df_transformed, results_df

def evaluate_metrics(y_true, y_pred, set_name="", fold=None, log_transformed=True):
    """Calculate multiple goodness of fit metrics"""
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)

    metrics = {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'ExplainedVar': explained_variance_score(y_true, y_pred)
    }
    fold_info = f" - Fold {fold}" if fold is not None else ""
    print(f"\n{set_name}{fold_info} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    return metrics
def handle_extreme_values(df):
    """Handle infinite and extreme values in the dataframe"""
    df_cleaned = df.copy()

    # Replace infinity with NaN
    df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)

    # For each numeric column
    for col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
        # Calculate Q1, Q3 and IQR
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # Cap extreme values
        df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

        # Fill NaN with median
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    return df_cleaned

# Data preparation
X_df = engineer_features(house_prices_train_data.drop(['SalePrice', 'Id'], axis=1))
y = house_prices_train_data['SalePrice']
y_log = np.log1p(y)  # Log transform target

# Handle missing values
for col in X_df.select_dtypes(include=['float64', 'int64']).columns:
    X_df[col] = X_df[col].fillna(X_df[col].mean())
for col in X_df.select_dtypes(include=['object']).columns:
    X_df[col] = X_df[col].fillna(X_df[col].mode()[0])

# Convert categorical variables to dummy variables
X_df = pd.get_dummies(X_df, drop_first=True)

# Handle extreme values before scaling
X_df = handle_extreme_values(X_df)

# Identify binary and continuous features
binary_features, continuous_features = analyze_feature_types(X_df)

# Store original X for feature importance calculation
X_original = X_df.copy()

# Scale features appropriately
scaler = StandardScaler()
X_scaled = X_df.copy()
X_scaled[continuous_features] = scaler.fit_transform(X_df[continuous_features])
# Data preparation
X_df = engineer_features(house_prices_train_data.drop(['SalePrice', 'Id'], axis=1))
y = house_prices_train_data['SalePrice']
y_log = np.log1p(y)  # Log transform target

# Handle missing values
for col in X_df.select_dtypes(include=['float64', 'int64']).columns:
    X_df[col] = X_df[col].fillna(X_df[col].mean())
for col in X_df.select_dtypes(include=['object']).columns:
    X_df[col] = X_df[col].fillna(X_df[col].mode()[0])

# Convert categorical variables to dummy variables
X_df = pd.get_dummies(X_df, drop_first=True)

# Handle extreme values before scaling
X_df = handle_extreme_values(X_df)

# Identify binary and continuous features
binary_features, continuous_features = analyze_feature_types(X_df)

# Store original X for feature importance calculation
X_original = X_df.copy()

# Scale features appropriately
scaler = StandardScaler()
X_scaled = X_df.copy()
X_scaled[continuous_features] = scaler.fit_transform(X_df[continuous_features])

# Perform dimension reduction using PCA
pca = PCA(n_components=0.95)  # Keep components that explain 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Print PCA results
print("\nPCA Analysis Results:")
print(f"Original dimensions: {X_scaled.shape[1]}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Explained variance ratio: {np.cumsum(pca.explained_variance_ratio_)[-1]:.4f}")

# Compare different reduction methods
reduced_datasets = compare_reduction_methods(X_scaled, y_log)

# Split data for shrinkage analysis
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_log, test_size=0.2, random_state=42
)
def train_with_shrinkage(X_train, y_train, X_test, y_test):
    """
    Train models with different shrinkage methods and evaluate their performance
    """
    # Ensure data is in the correct format
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
    # Create shrinkage models
    models = {
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42)
    }
    
    # Define alpha ranges for each model
    alphas = np.logspace(-4, 4, 20)  # Reduced number of alphas for speed
    
    results = {}
    evaluation = {}
    
    for name, model in models.items():
        try:
            # Grid search for best alpha
            grid = GridSearchCV(
                model,
                {'alpha': alphas},
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                error_score='raise'  # Added for debugging
            )
            
            # Fit the model
            grid.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'best_alpha': grid.best_params_['alpha'],
                'n_nonzero_coefs': np.sum(np.abs(grid.best_estimator_.coef_) > 1e-6)
            }
            
            # Make predictions
            train_pred = grid.predict(X_train)
            test_pred = grid.predict(X_test)
            
            # Calculate RMSE
            evaluation[name] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
            }
            
        except Exception as e:
            print(f"Error in {name} model: {str(e)}")
            results[name] = {
                'best_alpha': None,
                'n_nonzero_coefs': None
            }
            evaluation[name] = {
                'train_rmse': None,
                'test_rmse': None
            }
    
    return results, evaluation

# Then update the param_grids only for models that succeeded:
def update_param_grids(shrinkage_results):
    """Update param_grids with optimal alpha values from shrinkage analysis"""
    
    # Initialize default param_grids
    updated_param_grids = {
        'Linear': {},
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky']
        },
        'Lasso': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'selection': ['cyclic', 'random'],
            'max_iter': [2000]
        },
        'ElasticNet': {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [2000]
        },
        'Huber': {
            'epsilon': [1.1, 1.35, 1.5, 2.0],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [2000]
        }
    }
    
    # Update alphas only for models that succeeded
    for model_name in ['Ridge', 'Lasso', 'ElasticNet']:
        if (model_name in shrinkage_results and 
            shrinkage_results[model_name]['best_alpha'] is not None):
            
            optimal_alpha = shrinkage_results[model_name]['best_alpha']
            
            # Create range around optimal alpha
            alphas = np.logspace(
                np.log10(optimal_alpha/10), 
                np.log10(optimal_alpha*10), 
                5
            )
            
            # Update param_grid for this model
            updated_param_grids[model_name]['alpha'] = alphas.tolist()
    
    return updated_param_grids

# Use the functions:
# First perform shrinkage analysis
shrinkage_results, shrinkage_eval = train_with_shrinkage(X_train, y_train, X_test, y_test)

# Then update param_grids
param_grids = update_param_grids(shrinkage_results)

# Print results to verify
print("\nShrinkage Analysis Results:")
for name in shrinkage_results:
    if shrinkage_results[name]['best_alpha'] is not None:
        print(f"\n{name}:")
        print(f"Best alpha: {shrinkage_results[name]['best_alpha']}")
        print(f"Non-zero coefficients: {shrinkage_results[name]['n_nonzero_coefs']}")
        print(f"Train RMSE: {shrinkage_eval[name]['train_rmse']:.4f}")
        print(f"Test RMSE: {shrinkage_eval[name]['test_rmse']:.4f}")

print("\nUpdated Parameter Grids:")
for model, params in param_grids.items():
    if 'alpha' in params:
        print(f"\n{model} alphas:", params['alpha'])

# Update param_grids with optimal values
param_grids = update_param_grids(shrinkage_results)

print("\nUpdated Parameter Grids:")
for model, params in param_grids.items():
    if 'alpha' in params:
        print(f"\n{model} alphas:", params['alpha'])
# Perform shrinkage analysis
shrinkage_results, shrinkage_eval = train_with_shrinkage(X_train, y_train, X_test, y_test)

# Print shrinkage analysis results
print("\nShrinkage Analysis Results:")
print("-" * 50)
for model_name, results in shrinkage_results.items():
    print(f"\n{model_name}:")
    print(f"Best alpha: {results['best_alpha']}")
    print(f"Non-zero coefficients: {results['n_nonzero_coefs']}")
    print(f"Train RMSE: {shrinkage_eval[model_name]['train_rmse']:.4f}")
    print(f"Test RMSE: {shrinkage_eval[model_name]['test_rmse']:.4f}")

# Define number of folds for cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Visualize PCA results
plt.figure(figsize=(12, 4))

# Plot explained variance ratio
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance')

# Plot feature importance in first principal component
plt.subplot(1, 2, 2)
feature_importance = pd.DataFrame(
    pca.components_[0],
    columns=['Importance'],
    index=X_scaled.columns
)
feature_importance.sort_values('Importance', ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 Features in First Principal Component')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Create tuned models using GridSearchCV
tuned_models = {}
for name, model in base_models.items():
    if name in param_grids and param_grids[name]:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        tuned_models[name] = grid_search
    else:
        tuned_models[name] = model

# Initialize results dictionaries
model_results = {name: [] for name in tuned_models.keys()}
best_params = {}
fold_results = {name: [] for name in tuned_models.keys()}
# Model training and evaluation loop
for name, model in tuned_models.items():
    print(f"\n{'='*50}")
    print(f"Model: {name}")

    model_metrics = []

    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        X_fold_train, X_fold_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_fold_train, y_fold_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

        # Fit model
        model.fit(X_fold_train, y_fold_train)

        # Store best parameters if it's a GridSearchCV object
        if isinstance(model, GridSearchCV) and fold == 1:
            best_params[name] = model.best_params_
            print(f"\nBest parameters for {name}:")
            print(best_params[name])

        # Make predictions
        train_pred = model.predict(X_fold_train)
        val_pred = model.predict(X_fold_val)

        # Calculate metrics for this fold
        train_metrics = evaluate_metrics(y_fold_train, train_pred, "Training", fold)
        val_metrics = evaluate_metrics(y_fold_val, val_pred, "Validation", fold)

        # Store fold results
        fold_results[name].append({
            'fold': fold,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })

        # Store validation metrics for this fold
        model_metrics.append(val_metrics)

        # Create visualization plots for each fold
        fig = plt.figure(figsize=(15, 10))

        # Residual plots
        plt.subplot(2, 2, 1)
        plt.scatter(train_pred, train_pred - y_fold_train, alpha=0.5)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{name} - Training Residuals (Fold {fold})')
        plt.axhline(y=0, color='r', linestyle='--')

        plt.subplot(2, 2, 2)
        plt.scatter(val_pred, val_pred - y_fold_val, alpha=0.5)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{name} - Validation Residuals (Fold {fold})')
        plt.axhline(y=0, color='r', linestyle='--')

        # Feature importance plot
        importance_df = analyze_feature_importance(model, X_scaled.columns, binary_features, X_original)
        plt.subplot(2, 1, 2)
        sns.barplot(data=importance_df.head(15),
                   x='Feature',
                   y='Std_Impact',
                   hue='Type',
                   dodge=False)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{name} - Feature Importance (Fold {fold})')

        plt.tight_layout()
        plt.show()

    # Store average metrics for this model
    model_results[name] = model_metrics
    # Create comparison DataFrame
comparison_data = []
for name in model_results.keys():
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in model_results[name]])
        for metric in ['R2', 'RMSE', 'MAE', 'ExplainedVar']
    }
    std_metrics = {
        f"{metric}_std": np.std([fold[metric] for fold in model_results[name]])
        for metric in ['RMSE']
    }

    comparison_data.append({
        'Model': name,
        **avg_metrics,
        **std_metrics,
        'Best_Parameters': str(best_params.get(name, "No hyperparameters")),
        'Top_Features': ', '.join(analyze_feature_importance(tuned_models[name],
                                                         X_scaled.columns,
                                                         binary_features,
                                                         X_original)['Feature'].head(3).tolist())
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('RMSE', ascending=True)

# Plot comparison results
plt.figure(figsize=(15, 10))

# R² comparison
plt.subplot(2, 1, 1)
plt.bar(comparison_df['Model'], comparison_df['R2'])
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('Model Comparison - R² Score')
plt.xticks(rotation=45)

# RMSE comparison with error bars
plt.subplot(2, 1, 2)
plt.bar(comparison_df['Model'], comparison_df['RMSE'])
plt.errorbar(x=range(len(comparison_df)),
            y=comparison_df['RMSE'],
            yerr=comparison_df['RMSE_std'],
            fmt='none', color='black', capsize=5)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Model Comparison - RMSE with Standard Deviation')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# Replace everything after the model training loop with this:

# Initialize dictionary to store cumulative feature importance
cumulative_importance = pd.DataFrame(0, index=X_scaled.columns, 
                                   columns=['Importance', 'Frequency'])

# Combine feature importance across all folds
for name, fold_data in fold_results.items():
    for fold in fold_data:
        # Get feature importance for this fold
        fold_importance = analyze_feature_importance(tuned_models[name], 
                                                   X_scaled.columns,
                                                   binary_features,
                                                   X_original)
        
        # Add to cumulative importance
        cumulative_importance['Importance'] += fold_importance['Std_Impact']
        cumulative_importance['Frequency'] += (fold_importance['Std_Impact'] > 0).astype(int)

# Calculate average importance
cumulative_importance['Avg_Importance'] = cumulative_importance['Importance'] / n_folds
cumulative_importance['Selection_Frequency'] = cumulative_importance['Frequency'] / n_folds

# Sort by average importance
final_importance = cumulative_importance.sort_values('Avg_Importance', ascending=False)

# Print top features and their selection frequency
print("\nTop 20 Features Across All Folds:")
print(final_importance.head(20))

# Select features that appear in at least 80% of folds
consistent_features = final_importance[final_importance['Selection_Frequency'] >= 0.8].index.tolist()

print(f"\nNumber of consistently important features: {len(consistent_features)}")

# Prepare test data
test_df = engineer_features(house_prices_test_data.drop(['Id'], axis=1))

# Handle missing values in test data
for col in test_df.select_dtypes(include=['float64', 'int64']).columns:
    test_df[col] = test_df[col].fillna(test_df[col].mean())
for col in test_df.select_dtypes(include=['object']).columns:
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

# Convert categorical variables to dummy variables
test_df = pd.get_dummies(test_df, drop_first=True)

# Handle extreme values in test data
test_df = handle_extreme_values(test_df)

# Ensure test data has same columns as training data
missing_cols = set(X_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0
test_df = test_df[X_df.columns]

# Scale test data
test_scaled = test_df.copy()
test_scaled[continuous_features] = scaler.transform(test_df[continuous_features])

# Replace the final model training section with this:

# Build final model using consistent features
final_X = X_scaled[consistent_features]

# Get best model from cross-validation
best_model_name = comparison_df.iloc[0]['Model']

# Create a fresh instance of the best model with its best parameters
if best_model_name == 'Linear':
    final_model = LinearRegression()
elif best_model_name == 'Ridge':
    final_model = Ridge(
        alpha=tuned_models[best_model_name].best_params_['alpha'],
        random_state=42
    )
elif best_model_name == 'Lasso':
    final_model = Lasso(
        alpha=tuned_models[best_model_name].best_params_['alpha'],
        random_state=42
    )
elif best_model_name == 'ElasticNet':
    params = tuned_models[best_model_name].best_params_
    final_model = ElasticNet(
        alpha=params['alpha'],
        l1_ratio=params['l1_ratio'],
        random_state=42
    )
elif best_model_name == 'Huber':
    params = tuned_models[best_model_name].best_params_
    final_model = HuberRegressor(
        epsilon=params['epsilon'],
        alpha=params['alpha']
    )

# Convert data to numpy arrays and ensure they're 2D
final_X = np.asarray(final_X)
if len(final_X.shape) == 1:
    final_X = final_X.reshape(-1, 1)
y_log_array = np.asarray(y_log)

print(f"\nFinal model type: {type(final_model)}")
print(f"Final X shape: {final_X.shape}")
print(f"y_log shape: {y_log_array.shape}")

try:
    # Train final model
    final_model.fit(final_X, y_log_array)
    
    # Prepare test data with selected features
    test_final = test_scaled[consistent_features]
    test_final = np.asarray(test_final)
    if len(test_final.shape) == 1:
        test_final = test_final.reshape(-1, 1)
    
    # Make predictions
    final_predictions = np.expm1(final_model.predict(test_final))
    
    # Create final submission
    final_submission = pd.DataFrame({
        'Id': house_prices_test_data['Id'],
        'SalePrice': final_predictions
    })
    
    # Save submission file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_rmse = comparison_df.iloc[0]['RMSE']
    final_filename = f'submission_averaged_features_{best_model_name}_RMSE_{best_rmse:.4f}_{timestamp}.csv'
    final_path = os.path.join(submission_path, final_filename)
    final_submission.to_csv(final_path, index=False)
    
    print("\nModel training and prediction successful!")
    
except Exception as e:
    print(f"\nError during model training or prediction: {str(e)}")
    print("\nModel parameters:")
    print(final_model.get_params())
    print("\nData info:")
    print("X data types:", final_X.dtype)
    print("y data types:", y_log_array.dtype)
    print("Any NaN in X:", np.isnan(final_X).any())
    print("Any NaN in y:", np.isnan(y_log_array).any())
    raise
