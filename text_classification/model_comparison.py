import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch

# Import all models
from trainer import WrapperModel
from gru_trainer import GRUModel
from rnn_trainer import RNNModel
from lstm_trainer import LSTMModel
from tfidf_trainer import TFIDFClassifier, run_all_models
from data_prepare import train_dataloader, val_dataloader, num_classes


def compare_deep_learning_models():
    """So sánh các mô hình deep learning (BERT, GRU, RNN, LSTM)"""

    models_config = {
        'BERT': {
            'model_class': WrapperModel,
            'params': {'lr': 2e-5},  # num_labels tự động từ num_classes
            'max_epochs': 50
        },
        'GRU': {
            'model_class': GRUModel,
            # num_labels tự động
            'params': {'lr': 1e-3, 'embedding_dim': 128, 'hidden_dim': 256},
            'max_epochs': 100
        },
        'RNN': {
            'model_class': RNNModel,
            # num_labels tự động
            'params': {'lr': 1e-3, 'embedding_dim': 128, 'hidden_dim': 256},
            'max_epochs': 100
        },
        'LSTM': {
            'model_class': LSTMModel,
            # num_labels tự động
            'params': {'lr': 1e-3, 'embedding_dim': 128, 'hidden_dim': 256},
            'max_epochs': 100
        }
    }

    results = []

    for model_name, config in models_config.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name} Model")
        print(f"{'='*50}")

        try:
            # Initialize model
            model = config['model_class'](**config['params'])

            # Setup trainer
            logger = TensorBoardLogger(
                "tb_logs", name=f"{model_name.lower()}-comparison")
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=f'checkpoints/{model_name.lower()}',
                filename=f'best-{model_name.lower()}',
                save_last=True,
                mode='min',
            )

            trainer = Trainer(
                max_epochs=config['max_epochs'],
                gpus=1 if torch.cuda.is_available() else 0,
                logger=logger,
                callbacks=[
                    checkpoint_callback,
                    EarlyStopping(monitor='val_loss', patience=10)
                ],
                gradient_clip_algorithm='norm',
                gradient_clip_val=1.0,
                accumulate_grad_batches=2,
                min_epochs=20,
                enable_progress_bar=True
            )

            # Train model
            trainer.fit(model, train_dataloader, val_dataloader)

            # Get validation metrics
            val_results = trainer.validate(model, val_dataloader)

            results.append({
                'Model': model_name,
                'Val_Loss': val_results[0]['val_loss'],
                'Accuracy': val_results[0].get('accuracy', 0),
                'Macro_F1': val_results[0].get('macro_f1', 0),
                'Weighted_F1': val_results[0].get('weighted_f1', 0)
            })

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results.append({
                'Model': model_name,
                'Val_Loss': float('inf'),
                'Accuracy': 0,
                'Macro_F1': 0,
                'Weighted_F1': 0
            })

    return pd.DataFrame(results)


def create_comprehensive_comparison():
    """Tạo báo cáo so sánh toàn diện tất cả các mô hình"""

    print("Running Deep Learning Models Comparison...")
    dl_results = compare_deep_learning_models()

    print("\nRunning TF-IDF Models Comparison...")
    tfidf_results = run_all_models()

    # Convert TF-IDF results to DataFrame
    tfidf_df = []
    for model_type, result in tfidf_results.items():
        if result is not None:
            tfidf_df.append({
                'Model': f'TF-IDF + {model_type.upper()}',
                # Convert accuracy to loss-like metric
                'Val_Loss': 1 - result['accuracy'],
                'Accuracy': result['accuracy'],
                'Macro_F1': result['macro_f1'],
                'Weighted_F1': result['weighted_f1']
            })

    tfidf_df = pd.DataFrame(tfidf_df)

    # Combine results
    all_results = pd.concat([dl_results, tfidf_df], ignore_index=True)

    # Sort by accuracy
    all_results = all_results.sort_values('Accuracy', ascending=False)

    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL COMPARISON")
    print(f"{'='*80}")
    print(all_results.round(4))

    # Save results
    all_results.to_csv('comprehensive_model_comparison.csv', index=False)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Accuracy comparison
    plt.subplot(2, 2, 1)
    sns.barplot(data=all_results, x='Accuracy', y='Model', palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Accuracy')

    # Macro F1 comparison
    plt.subplot(2, 2, 2)
    sns.barplot(data=all_results, x='Macro_F1', y='Model', palette='plasma')
    plt.title('Macro F1 Score Comparison')
    plt.xlabel('Macro F1')

    # Weighted F1 comparison
    plt.subplot(2, 2, 3)
    sns.barplot(data=all_results, x='Weighted_F1',
                y='Model', palette='cividis')
    plt.title('Weighted F1 Score Comparison')
    plt.xlabel('Weighted F1')

    # Combined metrics heatmap
    plt.subplot(2, 2, 4)
    metrics_df = all_results.set_index(
        'Model')[['Accuracy', 'Macro_F1', 'Weighted_F1']]
    sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('All Metrics Heatmap')

    plt.tight_layout()
    plt.savefig('model_comparison_visualization.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    return all_results


def get_best_model_recommendation(results_df):
    """Đưa ra khuyến nghị mô hình tốt nhất"""

    best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
    best_macro_f1 = results_df.loc[results_df['Macro_F1'].idxmax()]
    best_weighted_f1 = results_df.loc[results_df['Weighted_F1'].idxmax()]

    print(f"\n{'='*60}")
    print("MODEL RECOMMENDATIONS")
    print(f"{'='*60}")

    print(
        f"🏆 Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    print(
        f"🏆 Best Macro F1: {best_macro_f1['Model']} ({best_macro_f1['Macro_F1']:.4f})")
    print(
        f"🏆 Best Weighted F1: {best_weighted_f1['Model']} ({best_weighted_f1['Weighted_F1']:.4f})")

    # Overall recommendation (weighted average)
    results_df['Overall_Score'] = (
        0.4 * results_df['Accuracy'] +
        0.3 * results_df['Macro_F1'] +
        0.3 * results_df['Weighted_F1']
    )

    best_overall = results_df.loc[results_df['Overall_Score'].idxmax()]
    print(
        f"\n🌟 Overall Best Model: {best_overall['Model']} (Score: {best_overall['Overall_Score']:.4f})")

    return {
        'best_accuracy': best_accuracy,
        'best_macro_f1': best_macro_f1,
        'best_weighted_f1': best_weighted_f1,
        'best_overall': best_overall
    }


if __name__ == "__main__":
    # Create directories
    import os
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('checkpoints/bert', exist_ok=True)
    os.makedirs('checkpoints/gru', exist_ok=True)
    os.makedirs('checkpoints/rnn', exist_ok=True)
    os.makedirs('checkpoints/lstm', exist_ok=True)

    # Run comprehensive comparison
    results = create_comprehensive_comparison()

    # Get recommendations
    recommendations = get_best_model_recommendation(results)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print("Check the following files for detailed results:")
    print("- comprehensive_model_comparison.csv")
    print("- model_comparison_visualization.png")
    print("- Individual model checkpoints in checkpoints/ folder")
    print(f"{'='*60}")
