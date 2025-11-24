class ModelManager:
    """Save and load trained models"""
    
    @staticmethod
    def save_models(rf_model, svm_model, dl_model, preprocessor, path='models/'):
        """Save all models"""
        os.makedirs(path, exist_ok=True)
        
        # Save Random Forest
        with open(f'{path}rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"[+] Random Forest saved to {path}rf_model.pkl")
        
        # Save SVM
        with open(f'{path}svm_model.pkl', 'wb') as f:
            pickle.dump(svm_model, f)
        print(f"[+] SVM saved to {path}svm_model.pkl")
        
        # Save Deep Learning model
        dl_model.model.save(f'{path}dl_model.h5')
        print(f"[+] Deep Learning model saved to {path}dl_model.h5")
        
        # Save preprocessor
        with open(f'{path}preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"[+] Preprocessor saved to {path}preprocessor.pkl")
    
    @staticmethod
    def load_models(path='models/'):
        """Load all models"""
        # Load Random Forest
        with open(f'{path}rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        print(f"[+] Random Forest loaded from {path}rf_model.pkl")
        
        # Load SVM
        with open(f'{path}svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        print(f"[+] SVM loaded from {path}svm_model.pkl")
        
        # Load Deep Learning model
        dl_model = keras.models.load_model(f'{path}dl_model.h5')
        print(f"[+] Deep Learning model loaded from {path}dl_model.h5")
        
        # Load preprocessor
        with open(f'{path}preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"[+] Preprocessor loaded from {path}preprocessor.pkl")
        
        return rf_model, svm_model, dl_model, preprocessor

# Save models
ModelManager.save_models(rf_detector, svm_detector, dl_detector, preprocessor)

# Load models (when needed)
# rf_loaded, svm_loaded, dl_loaded, prep_loaded = ModelManager.load_models()
