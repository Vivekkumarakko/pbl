class RandomForestDetector:
    """Random Forest for Evil Twin Detection"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train(self, X_train, y_train, tune_hyperparameters=True):
        """Train Random Forest model"""
        print("\n[*] Training Random Forest Classifier...")
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"[+] Best parameters: {self.best_params}")
        else:
            # Default parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        print(f"[+] Cross-validation F1 scores: {cv_scores}")
        print(f"[+] Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        print("\n=== Random Forest Performance ===")
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
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance

# Train Random Forest
rf_detector = RandomForestDetector()
rf_detector.train(X_train_balanced, y_train_balanced, tune_hyperparameters=False)
rf_results = rf_detector.evaluate(X_test, y_test)