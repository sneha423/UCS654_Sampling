"""
Five ML models optimized for imbalanced credit card fraud detection
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np


class ModelTrainer:
    def __init__(self):
        """Initialize all 5 models with optimal hyperparameters"""
        self.models = {
            'M1_LogisticRegression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='lbfgs',
                C=0.1
            ),
            'M2_RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'M3_SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'M4_XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                random_state=42,
                eval_metric='logloss'
            ),
            'M5_LightGBM': LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                is_unbalance=True,
                random_state=42,
                verbose=-1
            )
        }
        self.trained_models = {}
        self.model_scores = {}
    
    def train_model(self, model_name, X_train, y_train):
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model
    
    def train_all_models(self, X_train, y_train):
        """Train all 5 models"""
        print("[*] Training all models...")
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
            print(f"  ✓ {model_name} trained")
    
    def evaluate_model(self, model_name, X_train, y_train, X_test, y_test):
        """
        Evaluate model and detect overfitting
        Returns dict with train_acc, test_acc, generalization_gap, overfitting_status
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.trained_models[model_name]
        
        # Get predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate accuracy
        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)
        
        # Generalization gap
        gen_gap = train_acc - test_acc
        
        # Overfitting detection
        if gen_gap > 0.15:
            overfitting_status = "High"
        elif gen_gap > 0.08:
            overfitting_status = "Moderate"
        else:
            overfitting_status = "Low"
        
        result = {
            'model': model_name,
            'train_accuracy': round(train_acc, 4),
            'test_accuracy': round(test_acc, 4),
            'generalization_gap': round(gen_gap, 4),
            'overfitting_status': overfitting_status
        }
        
        self.model_scores[model_name] = result
        return result
    
    def evaluate_all_models(self, X_train, y_train, X_test, y_test):
        """Evaluate all trained models"""
        print("\n[*] Evaluating all models...")
        results = []
        
        for model_name in self.trained_models.keys():
            result = self.evaluate_model(model_name, X_train, y_train, X_test, y_test)
            results.append(result)
            print(f"  {model_name}:")
            print(f"    Train Acc: {result['train_accuracy']:.4f}")
            print(f"    Test Acc:  {result['test_accuracy']:.4f}")
            print(f"    Gen Gap:   {result['generalization_gap']:.4f}")
            print(f"    Overfitting: {result['overfitting_status']}")
        
        return results
    
    def get_best_model(self):
        """Get model with highest test accuracy"""
        if not self.model_scores:
            raise ValueError("No models have been evaluated yet")
        
        best = max(self.model_scores.items(), 
                  key=lambda x: x[1]['test_accuracy'])
        return best[0], best[1]['test_accuracy']
    
    def get_models_dict(self):
        """Get all model objects"""
        return self.models
    
    def get_trained_models(self):
        """Get all trained model objects"""
        return self.trained_models


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, weights=[0.9, 0.1], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train)
    trainer.evaluate_all_models(X_train, y_train, X_test, y_test)
    
    best_model, best_acc = trainer.get_best_model()
    print(f"\nBest Model: {best_model} (Accuracy: {best_acc:.4f})")