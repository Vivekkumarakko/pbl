class HybridEnsembleDetector:
    """Hybrid ensemble combining RF, SVM, and DL"""
    
    def __init__(self, rf_model, svm_model, dl_model):
        self.rf_model = rf_model
        self.svm_model = svm_model
        self.dl_model = dl_model
        self.weights = [0.35, 0.30, 0.35]  # RF, SVM, DL
        
    def predict_proba(self, X):
        """Weighted ensemble prediction"""
        # Get probabilities from each model
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        svm_proba = self.svm_model.predict_proba(X)[:, 1]
        dl_proba = self.dl_model.predict_proba(X)
        
        # Weighted average
        ensemble_proba = (
            self.weights[0] * rf_proba +
            self.weights[1] * svm_proba +
            self.weights[2] * dl_proba
        )
        
        return ensemble_proba
    
    def predict(self, X, threshold=0.5):
        """Make final predictions"""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        print("\n=== Hybrid Ensemble Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'roc_auc': roc_auc
        }

# Create ensemble
ensemble_detector = HybridEnsembleDetector(rf_detector, svm_detector, dl_detector)
ensemble_results = ensemble_detector.evaluate(X_test, y_test)