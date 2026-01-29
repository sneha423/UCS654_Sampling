"""
Main execution script - Orchestrates entire sampling and model evaluation pipeline
Handles all 5 sampling techniques on all 5 ML models
Generates final summary table
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from sampling_techniques import SamplingTechniques
from models import ModelTrainer


class PipelineOrchestrator:
    def __init__(self, dataset_path):
        """Initialize pipeline"""
        self.dataset_path = dataset_path
        self.results = []
        self.detailed_results = {}
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        print("=" * 70)
        print("IMBALANCED DATASET SAMPLING & MODEL EVALUATION PIPELINE")
        print("=" * 70)
        
        # Step 1: Load and preprocess data
        print("\n[STEP 1] DATA LOADING & PREPROCESSING")
        print("-" * 70)
        loader = DataLoader(self.dataset_path)
        loader.load_data()
        loader.check_imbalance()
        X_train, X_test, y_train, y_test = loader.preprocess()
        
        # Step 2: Generate 5 samples using different techniques
        print("\n[STEP 2] GENERATING SAMPLES USING 5 TECHNIQUES")
        print("-" * 70)
        sampler = SamplingTechniques(X_train, y_train)
        
        samples = {
            'Sampling1': sampler.simple_random_sampling(),
            'Sampling2': sampler.stratified_random_sampling(),
            'Sampling3': sampler.cluster_sampling(),
            'Sampling4': sampler.systematic_sampling(),
            'Sampling5_KFold': sampler.kfold_sampling()
        }
        
        print("[✓] Generated samples:")
        print("  - Sampling1: Simple Random Sampling")
        print("  - Sampling2: Stratified Random Sampling")
        print("  - Sampling3: Cluster Sampling")
        print("  - Sampling4: Systematic Sampling")
        print("  - Sampling5: K-Fold Cross-Validation")
        
        # Step 3: Train and evaluate models on each sample
        print("\n[STEP 3] TRAINING & EVALUATING MODELS ON EACH SAMPLE")
        print("-" * 70)
        
        sample_results = {}
        
        # Handle K-Fold separately (multiple folds)
        kfold_results = []
        kfold_data = samples.pop('Sampling5_KFold')
        
        # Process regular samples (1-4)
        for sample_id, (sample_idx, (X_sample, y_sample, _)) in enumerate(
            list(samples.items()), 1
        ):
            print(f"\n[Sampling{sample_id}] {sample_idx}")
            print(f"  Sample shape: {X_sample.shape}")
            
            # Train models
            trainer = ModelTrainer()
            trainer.train_all_models(X_sample, y_sample)
            
            # Evaluate on test set
            eval_results = trainer.evaluate_all_models(X_test, y_test, X_test, y_test)
            
            # Find best model for this sample
            best_model = max(eval_results, key=lambda x: x['test_accuracy'])
            
            print(f"\n  Best Model: {best_model['model']}")
            print(f"  Best Accuracy: {best_model['test_accuracy']:.4f}")
            
            # Store result
            result_record = {
                'Sample_ID': f'Sample_{sample_id}',
                'Sampling_Technique': sample_idx,
                'Best_Model': best_model['model'],
                'Accuracy': best_model['test_accuracy'],
                'Generalization_Gap': best_model['generalization_gap'],
                'Overfitting_Status': best_model['overfitting_status']
            }
            self.results.append(result_record)
            sample_results[f'Sampling{sample_id}'] = (best_model['model'], best_model['test_accuracy'])
        
        # Process K-Fold (multiple folds)
        print(f"\n[Sampling5] K-Fold Cross-Validation")
        print(f"  Processing {len(kfold_data)} folds...")
        
        fold_accuracies = {model_name: [] for model_name in [
            'M1_LogisticRegression', 'M2_RandomForest', 'M3_SVM', 'M4_XGBoost', 'M5_LightGBM'
        ]}
        
        for fold_idx, fold in enumerate(kfold_data, 1):
            X_train_fold, y_train_fold = fold['train']
            X_test_fold, y_test_fold = fold['test']
            
            trainer = ModelTrainer()
            trainer.train_all_models(X_train_fold, y_train_fold)
            eval_results = trainer.evaluate_all_models(X_train_fold, y_train_fold, X_test_fold, y_test_fold)
            
            for result in eval_results:
                fold_accuracies[result['model']].append(result['test_accuracy'])
        
        # Calculate average accuracies across folds
        avg_accuracies = {
            model: np.mean(accs) for model, accs in fold_accuracies.items()
        }
        best_kfold_model = max(avg_accuracies, key=avg_accuracies.get)
        best_kfold_acc = avg_accuracies[best_kfold_model]
        
        print(f"\n  Best Model (across folds): {best_kfold_model}")
        print(f"  Average Accuracy: {best_kfold_acc:.4f}")
        
        result_record = {
            'Sample_ID': 'Sample_5',
            'Sampling_Technique': 'Sampling5_KFold',
            'Best_Model': best_kfold_model,
            'Accuracy': best_kfold_acc,
            'Generalization_Gap': 'N/A (Cross-validation)',
            'Overfitting_Status': 'Controlled'
        }
        self.results.append(result_record)
        
        # Step 4: Generate summary table
        print("\n[STEP 4] GENERATING SUMMARY TABLE")
        print("-" * 70)
        
        results_df = pd.DataFrame(self.results)
        print("\n" + results_df.to_string(index=False))
        
        # Save to CSV
        results_df.to_csv('results.csv', index=False)
        print("\n[✓] Results saved to 'results.csv'")
        
        # Step 5: Analysis and insights
        print("\n[STEP 5] ANALYSIS & INSIGHTS")
        print("-" * 70)
        self.print_insights(results_df)
        
        return results_df
    
    def print_insights(self, results_df):
        """Print analysis and insights"""
        print("\n📊 SUMMARY OF FINDINGS:")
        print("-" * 70)
        
        # Best overall model
        best_overall = results_df.loc[results_df['Accuracy'].idxmax()]
        print(f"\n1. Best Overall Performance:")
        print(f"   - Sample: {best_overall['Sample_ID']}")
        print(f"   - Technique: {best_overall['Sampling_Technique']}")
        print(f"   - Model: {best_overall['Best_Model']}")
        print(f"   - Accuracy: {best_overall['Accuracy']:.4f}")
        
        # Model frequency
        print(f"\n2. Model Performance Distribution:")
        model_counts = results_df['Best_Model'].value_counts()
        for model, count in model_counts.items():
            print(f"   - {model}: {count} samples")
        
        # Overfitting analysis
        print(f"\n3. Overfitting Status:")
        if 'Overfitting_Status' in results_df.columns:
            overfit_counts = results_df['Overfitting_Status'].value_counts()
            for status, count in overfit_counts.items():
                print(f"   - {status}: {count} samples")
        
        # Sampling technique effectiveness
        print(f"\n4. Sampling Technique Effectiveness:")
        tech_accuracy = results_df.groupby('Sampling_Technique')['Accuracy'].mean()
        for tech, acc in tech_accuracy.items():
            print(f"   - {tech}: {acc:.4f}")
        
        print("\n" + "=" * 70)


def main():
    """Main entry point"""
    # Dataset path - update with your actual path
    dataset_path = "Creditcard_data.csv"
    
    try:
        orchestrator = PipelineOrchestrator(dataset_path)
        results_df = orchestrator.run_pipeline()
        
        print("\n✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("\nNext Steps:")
        print("1. Review results.csv for final summary table")
        print("2. Push code to GitHub with discussion of findings")
        print("3. Include analysis of which sampling techniques work best")
        
    except FileNotFoundError:
        print(f"❌ Error: Dataset file '{dataset_path}' not found")
        print(f"Please download from: https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv")
    except Exception as e:
        print(f"❌ Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()