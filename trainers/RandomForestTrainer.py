# Import all necessary libraries
import centralLogging as cl
logger = cl.get_logger(console_level="INFO", file_level="DEBUG")
logger.info("="*30)
logger.info("RANDOM FOREST TRAINER")
logger.info("="*30)

try:
    import pandas as pd
    import numpy as np
    import warnings
    from pathlib import Path
    import time
    from datetime import datetime
    import itertools
    import pickle
    import joblib
    import json
    import os
    import argparse

    # Machine Learning libraries
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, label_binarize
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        precision_recall_curve, roc_curve, matthews_corrcoef,
        cohen_kappa_score, balanced_accuracy_score
    )
    from sklearn.calibration import calibration_curve
    from sklearn.ensemble import RandomForestClassifier

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils.filehash import get_file_hash
    logger.debug("All libraries imported successfully.")
except ImportError as e:
    logger.error(f"Error importing libraries: {e}")
    exit(1)

def load_dataset(file_path):
    """Load dataset from a CSV file."""
    logger.info(f"Loading dataset from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        exit(1)
        
def drop_unnecessary_columns(df):
    """Drop all columns except 'label' and features."""
    logger.info("=== Dropping unnecessary columns ===")
    try:
        columns_to_keep = ['label'] + ['class'] + [col for col in df.columns if col.startswith('feature_')]
        columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
        original_cols = df.shape[1]
        df = df.drop(columns=columns_to_drop, errors='ignore')
        dropped = original_cols - df.shape[1]
        if dropped > 0:
            logger.info(f"Dropped {dropped} columns: {columns_to_drop}")
        else:
            logger.warning("No columns dropped.")
            
        logger.info(f"Total columns: {df.shape[1]}")
        return df
    except Exception as e:
        logger.error(f"Error dropping columns: {e}")
        exit(1)

def split_data(df, test_size=0.2, random_state=420):
    """Split the dataset into training and testing sets."""
    logger.info("=== Splitting dataset into training and testing sets ===")
    try:
        X = df.drop(columns=['label', 'class'], errors='ignore')
        y = df['label'] if 'label' in df.columns else df['class']
        
        logger.info(f"Target distribution: {y.value_counts()}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        logger.info(f"Dataset split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    except KeyError as e:
        logger.error(f"Missing expected column: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        exit(1)
        
def parameter_setup():
    """Define hyperparameters for Random Forest."""
    
    '''
    param_grid_1 = {
        'n_estimators': [100, 200, 300],
        'max_depth': ["None", 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    '''
    '''
    param_grid_2 = {
        'n_estimators': [150, 200, 250],
        'max_depth': [None, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 4, 6],
        'max_features': [None, 0.2, 0.4, 'sqrt'],
        'bootstrap': [False]
    }
    '''
    
    param_grid = {
        'n_estimators': [150, 175, 200],
        'max_depth': [None],
        'min_samples_split': [5],
        'min_samples_leaf': [1],
        'max_features': [0.1, 0.2, 0.3, 0.4],
        'bootstrap': [False]
    }
    
    # Create all combinations of hyperparameters
    param_combinations = list(itertools.product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['min_samples_split'],
        param_grid['min_samples_leaf'],
        param_grid['max_features'],
        param_grid['bootstrap']
    ))
    
    logger.debug(f"Parameter grid: {param_grid}")

    logger.info(f"Total hyperparameter combinations to try: {len(param_combinations)}")
    
    return param_combinations, param_grid

def train_random_forest(X_train, X_test, y_train, y_test, param_combinations):
    logger.info("=== Starting Random Forest training ===")
    results = []
    best_score = 0
    best_model = None
    best_params = None

    try:
        for idx, params in enumerate(param_combinations):
            n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap = params
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=420,
                n_jobs=-2,
                verbose=1
            )

            try:
                logger.info(f"Training model #{idx+1}/{len(param_combinations)}")
                model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Error training model with params {params}: {e}")
                continue
            
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'params': params,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1
            })
            
            logger.info(f"[{idx+1}/{len(param_combinations)}]\tAccuracy: {accuracy_score(y_test, y_pred):.4f} | Precision: {precision_score(y_test, y_pred, average='weighted'):.4f} | Recall: {recall_score(y_test, y_pred, average='weighted'):.4f} | F1 Score: {f1:.4f} | Params: {params}")

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_params = params
                logger.info(f"Best Model Updated | F1: {best_score}")
            else:
                logger.debug(f"No improvement. Current best F1: {best_score} for params: {best_params}")
    finally:
        return best_model, best_params, results

def plot_roc_curves(best_model, X_test, y_test, outputDir):
    classes = best_model.classes_
    y_prob  = best_model.predict_proba(X_test)
    y_bin   = label_binarize(y_test, classes=classes)
    
    if len(classes) == 2:
        # Column 0 is negative (1 - y_bin), Column 1 is positive (y_bin)
        y_bin = np.hstack([1 - y_bin, y_bin])

    plt.figure(figsize=(10, 7))
    plt.title("ROC Curves (Per Class)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_score   = roc_auc_score(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{outputDir}/roc_curves.png")
    plt.show()
    plt.close()
    logger.info(f"ROC curves saved to {outputDir}/roc_curves.png")
    
def plot_precision_recall_curves(best_model, X_test, y_test, outputDir):
    classes = best_model.classes_
    y_prob  = best_model.predict_proba(X_test)
    y_bin   = label_binarize(y_test, classes=classes)
    
    if len(classes) == 2:
        # Column 0 is negative (1 - y_bin), Column 1 is positive (y_bin)
        y_bin = np.hstack([1 - y_bin, y_bin])

    plt.figure(figsize=(10, 7))
    plt.title("Precision-Recall Curves (Per Class)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f"Class {cls}")

    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{outputDir}/precision_recall_curves.png")
    plt.show()
    plt.close()
    logger.info(f"PR curves saved to {outputDir}/precision_recall_curves.png")
    
def plot_calibration_curves(best_model, X_test, y_test, outputDir):
    classes = best_model.classes_
    y_prob  = best_model.predict_proba(X_test)
    y_bin   = label_binarize(y_test, classes=classes)
    
    if len(classes) == 2:
        # Column 0 is negative (1 - y_bin), Column 1 is positive (y_bin)
        y_bin = np.hstack([1 - y_bin, y_bin])

    plt.figure(figsize=(10, 7))
    plt.title("Calibration Curves (Per Class)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")

    for i, cls in enumerate(classes):
        fraction_pos, mean_pred = calibration_curve(y_bin[:, i], y_prob[:, i], n_bins=10)
        plt.plot(mean_pred, fraction_pos, marker='o', label=f"Class {cls}")

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{outputDir}/calibration_curves.png")
    plt.show()
    plt.close()
    logger.info(f"Calibration curves saved to {outputDir}/calibration_curves.png")

def save_results(results, param_grid, best_params, outputDir):
    logger.info("=== Saving results ===")
    """Save results to a JSON file."""
    try:
        output_path = os.path.join(outputDir, "random_forest_results.json")
        with open(output_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'param_grid': param_grid,
                'results': results
            }, f, indent=4)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        
def save_best_model(best_model, outputDir):
    logger.info("=== Saving best model ===")
    """Save the best model using joblib."""
    try:
        model_path = os.path.join(outputDir, "best_random_forest_model.joblib")
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        model_hash = get_file_hash(model_path)
        logger.info(f"Best Model's SHA256 Hash: {model_hash}")
        hash_output_path = f"{model_path}.sha256"
        
        try:
            with open(hash_output_path, "w") as f:
                f.write(model_hash)
            logger.info(f"Model hash saved to: {hash_output_path}")
        except Exception as e:
            logger.error(f"Error saving model hash: {e}")

    except Exception as e:
        logger.error(f"Error saving best model: {e}")

def feature_importance(best_model, X, outputDir):
    logger.info("=== Plotting feature importance ===")
    """Plot feature importance."""
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(20)

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
    plt.tight_layout()
    plt.savefig(f"{outputDir}/feature_importance.png")
    plt.show()
    plt.close()
    
def best_model_analysis(best_model, X_test, y_test, outputDir):
    logger.info("=== Analyzing best model ===")
    """Analyze the best model's performance."""
    try:
        y_pred = best_model.predict(X_test)
    except Exception as e:
        logger.error(f"Error predicting with best model: {e}")
        return
    
    try:
        y_prob = best_model.predict_proba(X_test)
    except Exception as e:
        logger.error(f"Error predicting probabilities with best model: {e}")
        y_prob = None

    # Classification report
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")
        
    try:
        mcc = matthews_corrcoef(y_test, y_pred)
    except Exception as e:
        logger.error(f"Error calculating MCC: {e}")
        mcc = 'N/A'
    
    try:
        kappa = cohen_kappa_score(y_test, y_pred)
    except Exception as e:
        logger.error(f"Error calculating Cohen's Kappa: {e}")
        kappa = 'N/A'
    
    try:
        bal_acc = balanced_accuracy_score(y_test, y_pred)
    except Exception as e:
        logger.error(f"Error calculating Balanced Accuracy: {e}")
        bal_acc = 'N/A'

    try:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted') if y_prob is not None else 'N/A'
    except Exception as e:
        logger.error(f"Error calculating ROC-AUC: {e}")
        roc_auc = 'N/A'

    logger.info(f"MCC:\t{mcc}")
    logger.info(f"Cohen's Kappa:\t{kappa}")
    logger.info(f"Balanced Accuracy:\t{bal_acc}")
    logger.info(f"ROC-AUC (weighted ovr):\t{roc_auc}")
    
    logger.info(f"Saving classification report to {outputDir}/classification_report.txt")
    with open(f"{outputDir}/classification_report.txt", 'w') as f:
        f.write(report)
        f.write(f"\nMCC:\t{mcc}")
        f.write(f"\nCohen's Kappa:\t{kappa}")
        f.write(f"\nBalanced Accuracy:\t{bal_acc}")
        f.write(f"\nROC-AUC (weighted ovr):\t{roc_auc}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.tight_layout()
    plt.savefig(f"{outputDir}/confusion_matrix.png")
    plt.show()
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(
        description="Random Forest Trainer for Deepfake Detection",
        formatter_class=argparse.RawTextHelpFormatter
        )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to the input dataset CSV file"
        )
    parser.add_argument(
        '--output',
        type=str,
        default="results/images/random_forest",
        help="Directory to save results and models"
        )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)"
        )
    parser.add_argument(
        '--random_state',
        type=int,
        default=420,
        help="Random state for reproducibility (default: 420)"
        )
    args = parser.parse_args()
    
    #test_size = args.test_size
    #random_state = args.random_state
    #file_path = args.input
    outputDir = args.output
    
    # Create output directory if it doesn't exist
    os.makedirs(outputDir, exist_ok=True)
    
    df = load_dataset(args.input)
    df = drop_unnecessary_columns(df)
    
    X_train, X_test, y_train, y_test = split_data(df, test_size=args.test_size, random_state=args.random_state)
    
    param_combinations, param_grid = parameter_setup()
    
    best_model, best_params, results = None, None, []
    try:
        best_model, best_params, results = train_random_forest(X_train, X_test, y_train, y_test, param_combinations)
        
        save_results(results, param_grid, best_params, outputDir)
        save_best_model(best_model, outputDir)
        best_model_analysis(best_model, X_test, y_test, outputDir)
        
        try:
            plot_roc_curves(best_model, X_test, y_test, outputDir)
        except Exception as e:
            logger.error(f"Error plotting ROC curves: {e}")
        try:
            plot_precision_recall_curves(best_model, X_test, y_test, outputDir)
        except Exception as e:
            logger.error(f"Error plotting Precision-Recall curves: {e}")
        try:
            plot_calibration_curves(best_model, X_test, y_test, outputDir)
        except Exception as e:
            logger.error(f"Error plotting Calibration curves: {e}")
        try:
            feature_importance(best_model, X_train, outputDir)
        except Exception as e:
            logger.error(f"Error plotting Feature Importance: {e}")
    except KeyboardInterrupt:
        logger.warning("!! Training interrupted by user !!")
        
        try:
            save_results(results, param_grid, best_params, outputDir)
            save_best_model(best_model, outputDir)
            best_model_analysis(best_model, X_test, y_test, outputDir)
        except (best_params is None or best_model is None) as e:
            logger.warning("No model trained yet, skipping save and analysis: {e}")
            exit(0)
        except Exception as e:
            logger.error(f"Error during save/analysis after interruption: {e}")
            exit(1)
        
        try:
            plot_roc_curves(best_model, X_test, y_test, outputDir)
        except Exception as e:
            logger.error(f"Error plotting ROC curves: {e}")
        try:
            plot_precision_recall_curves(best_model, X_test, y_test, outputDir)
        except Exception as e:
            logger.error(f"Error plotting Precision-Recall curves: {e}")
        try:
            plot_calibration_curves(best_model, X_test, y_test, outputDir)
        except Exception as e:
            logger.error(f"Error plotting Calibration curves: {e}")
        try:
            feature_importance(best_model, X_train, outputDir)
        except Exception as e:
            logger.error(f"Error plotting Feature Importance: {e}")
    finally:
        logger.info("=== Random Forest training completed ===")
    
if __name__ == "__main__":
    main()