def main():
    """Main execution pipeline"""
    print("="*60)
    print("EVIL TWIN ATTACK DETECTION SYSTEM")
    print("Machine Learning-Based Framework")
    print("="*60)
    
    # Step 1: Generate/Load Dataset
    print("\n[1] Generating Dataset...")
    dataset = generate_synthetic_dataset(n_samples=2000)
    print(f"    Dataset shape: {dataset.shape}")
    
    # Step 2: Preprocess Data
    print("\n[2] Preprocessing Data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(dataset, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(
        X_train, y_train, strategy='SMOTE'
    )
    print(f"    Training samples: {X_train_balanced.shape[0]}")
    print(f"    Test samples: {X_test.shape[0]}")
    
    # Step 3: Train Random Forest
    print("\n[3] Training Random Forest...")
    rf_detector = RandomForestDetector()
    rf_detector.train(X_train_balanced, y_train_balanced, tune_hyperparameters=False)
    rf_results = rf_detector.evaluate(X_test, y_test)
    
    # Step 4: Train SVM
    print("\n[4] Training SVM...")
    svm_detector = SVMDetector()
    svm_detector.train(X_train_balanced, y_train_balanced, tune_hyperparameters=False)
    svm_results = svm_detector.evaluate(X_test, y_test)
    
    # Step 5: Train Deep Learning
    print("\n[5] Training Deep Learning Model...")
    dl_detector = DeepLearningDetector(input_dim=X_train.shape[1])
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
        X_train_balanced, y_train_balanced, test_size=0.2, random_state=42
    )
    dl_detector.train(X_train_dl, y_train_dl, X_val_dl, y_val_dl, epochs=50)
    dl_results = dl_detector.evaluate(X_test, y_test)
    
    # Step 6: Create Ensemble
    print("\n[6] Creating Hybrid Ensemble...")
    ensemble_detector = HybridEnsembleDetector(rf_detector, svm_detector, dl_detector)
    ensemble_results = ensemble_detector.evaluate(X_test, y_test)
    
    # Step 7: Visualize Results
    print("\n[7] Generating Visualizations...")
    visualizer = ModelVisualizer()
    all_results = {
        'Random Forest': rf_results,
        'SVM': svm_results,
        'Deep Learning': dl_results,
        'Hybrid Ensemble': ensemble_results
    }
    visualizer.plot_roc_curves(all_results, y_test)
    visualizer.plot_confusion_matrices(all_results, y_test)
    metrics_df = visualizer.plot_performance_comparison(all_results, y_test)
    
    # Step 8: Save Models
    print("\n[8] Saving Models...")
    ModelManager.save_models(rf_detector, svm_detector, dl_detector, preprocessor)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📊 Final Performance Summary:")
    print(metrics_df.to_string(index=False))
    
    return {
        'models': (rf_detector, svm_detector, dl_detector, ensemble_detector),
        'results': all_results,
        'preprocessor': preprocessor
    }

# Run the complete pipeline
if __name__ == "__main__":
    experiment_results = main()