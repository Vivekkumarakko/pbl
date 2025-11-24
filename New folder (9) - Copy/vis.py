class ModelVisualizer:
    """Visualize model performance"""
    
    @staticmethod
    def plot_roc_curves(models_results, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in models_results.items():
            fpr, tpr, _ = roc_curve(y_test, results['y_proba'])
            auc_score = results['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Evil Twin Detection Models', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrices(models_results, y_test):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(models_results.items()):
            cm = confusion_matrix(y_test, results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Legitimate', 'Evil Twin'],
                       yticklabels=['Legitimate', 'Evil Twin'])
            axes[idx].set_title(f'{model_name} - Confusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_performance_comparison(models_results, y_test):
        """Compare model performance metrics"""
        metrics_data = []
        
        for model_name, results in models_results.items():
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, results['y_pred'], average='binary'
            )
            metrics_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC AUC': results['roc_auc']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_metrics))
        width = 0.15
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, metric in enumerate(metrics):
            ax.bar(x + idx*width, df_metrics[metric], width, 
                  label=metric, color=colors[idx])
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_metrics['Model'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_metrics
    
    @staticmethod
    def plot_feature_importance(rf_model, feature_names):
        """Plot feature importance from Random Forest"""
        importance_df = rf_model.get_feature_importance(feature_names)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature', palette='viridis')
        plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

# Visualize results
visualizer = ModelVisualizer()

# Compile all results
all_results = {
    'Random Forest': rf_results,
    'SVM': svm_results,
    'Deep Learning': dl_results,
    'Hybrid Ensemble': ensemble_results
}

# Generate visualizations
visualizer.plot_roc_curves(all_results, y_test)
visualizer.plot_confusion_matrices(all_results, y_test)
metrics_df = visualizer.plot_performance_comparison(all_results, y_test)

# Feature importance
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
visualizer.plot_feature_importance(rf_detector, feature_names)

print("\n=== Final Performance Summary ===")
print(metrics_df.to_string(index=False))