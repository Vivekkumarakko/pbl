class DeepLearningDetector:
    """Deep Neural Network for Evil Twin Detection"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build DNN architecture"""
        model = Sequential([
            Dense(128, activation='relu', input_dim=self.input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train deep learning model"""
        print("\n[*] Training Deep Learning Model...")
        
        # Build model
        self.build_model()
        print(self.model.summary())
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        print("[+] Training completed!")
        
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict(X).flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        print("\n=== Deep Learning Performance ===")
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
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

# Train Deep Learning Model
dl_detector = DeepLearningDetector(input_dim=X_train.shape[1])

# Create validation set
X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
    X_train_balanced, y_train_balanced, test_size=0.2, random_state=42
)

dl_detector.train(X_train_dl, y_train_dl, X_val_dl, y_val_dl, epochs=50, batch_size=32)
dl_results = dl_detector.evaluate(X_test, y_test)
dl_detector.plot_training_history()