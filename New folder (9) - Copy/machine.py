class DataPreprocessor:
    """Preprocess data for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess(self, df, fit=True):
        """Preprocess the dataset"""
        df_processed = df.copy()
        
        # Separate features and labels
        if 'label' in df_processed.columns:
            X = df_processed.drop('label', axis=1)
            y = df_processed['label']
        else:
            X = df_processed
            y = None
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def handle_imbalance(self, X, y, strategy='SMOTE'):
        """Handle class imbalance"""
        if strategy == 'SMOTE':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif strategy == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
            
        return X_resampled, y_resampled

# Preprocess data
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess(dataset, fit=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Handle imbalance
X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(
    X_train, y_train, strategy='SMOTE'
)
print(f"Balanced training set size: {X_train_balanced.shape}")