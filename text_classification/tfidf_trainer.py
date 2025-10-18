import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import catboost as cb
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pytorch_lightning import seed_everything
from data_prepare import num_classes
import warnings
warnings.filterwarnings('ignore')

seed_everything(42)


class TFIDFClassifier:
    def __init__(self, model_type='logistic', max_features=10000, ngram_range=(1, 2)):
        self.model_type = model_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # Vietnamese doesn't have built-in stop words
            lowercase=True,
            strip_accents='unicode'
        )
        self.model = None
        self.class_weights = None

    def _get_model(self):
        if self.model_type == 'logistic':
            return LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(
                class_weight='balanced',
                random_state=42,
                probability=True
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB()
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                objective='multiclass',
                num_class=num_classes,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif self.model_type == 'catboost':
            return cb.CatBoostClassifier(
                iterations=1000,
                learning_rate=0.1,
                depth=6,
                class_weights='Balanced',
                random_seed=42,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def prepare_data(self):
        # Load dataset
        raw = load_dataset(
            'csv',
            data_files=r"C:\Users\Vitus\Downloads\UIT-VSFC_train.csv"
        )

        dataset = raw['train'].train_test_split(test_size=0.2, seed=42)

        # Extract text and labels
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        val_texts = dataset['test']['text']
        val_labels = dataset['test']['label']

        return train_texts, train_labels, val_texts, val_labels

    def fit(self, train_texts, train_labels):
        print(f"Training {self.model_type} model...")

        # Fit TF-IDF vectorizer and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(train_texts)

        # Compute class weights
        unique_labels = np.unique(train_labels)
        self.class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=train_labels
        )

        # Initialize and train model
        self.model = self._get_model()

        if self.model_type == 'naive_bayes':
            # Naive Bayes doesn't support class_weight, so we'll use sample_weight
            sample_weights = np.array(
                [self.class_weights[label] for label in train_labels])
            self.model.fit(X_train_tfidf, train_labels,
                           sample_weight=sample_weights)
        else:
            self.model.fit(X_train_tfidf, train_labels)

        print(f"Training completed for {self.model_type}")

    def predict(self, texts):
        X_tfidf = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_tfidf)
        probabilities = self.model.predict_proba(X_tfidf)
        return predictions, probabilities

    def evaluate(self, val_texts, val_labels):
        print(f"Evaluating {self.model_type} model...")

        predictions, probabilities = self.predict(val_texts)

        # Calculate metrics
        accuracy = accuracy_score(val_labels, predictions)
        report = classification_report(
            val_labels, predictions, output_dict=True, zero_division=0)
        cm = confusion_matrix(val_labels, predictions)

        # Print results
        print(f"\n{self.model_type.upper()} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(
            f'confusion_matrix_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'accuracy': accuracy,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'weighted_precision': report['weighted avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'weighted_recall': report['weighted avg']['recall'],
            'confusion_matrix': cm,
            'classification_report': report
        }

    def save_model(self, filepath):
        """Save the trained model and vectorizer"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'class_weights': self.class_weights
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model and vectorizer"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.class_weights = model_data['class_weights']
        print(f"Model loaded from {filepath}")


def run_all_models():
    """Run all TF-IDF based models and compare results"""

    models = [
        'logistic',
        'random_forest',
        'svm',
        'naive_bayes',
        'lightgbm',
        'catboost'
    ]

    results = {}

    # Prepare data once
    classifier = TFIDFClassifier()
    train_texts, train_labels, val_texts, val_labels = classifier.prepare_data()

    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Running {model_type.upper()} Model")
        print(f"{'='*50}")

        try:
            # Initialize classifier
            clf = TFIDFClassifier(model_type=model_type)

            # Train model
            clf.fit(train_texts, train_labels)

            # Evaluate model
            result = clf.evaluate(val_texts, val_labels)
            results[model_type] = result

            # Save model
            clf.save_model(f'models/tfidf_{model_type}_model.joblib')

        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = None

    # Compare results
    print(f"\n{'='*50}")
    print("MODEL COMPARISON")
    print(f"{'='*50}")

    comparison_df = []
    for model_type, result in results.items():
        if result is not None:
            comparison_df.append({
                'Model': model_type.upper(),
                'Accuracy': result['accuracy'],
                'Macro F1': result['macro_f1'],
                'Weighted F1': result['weighted_f1'],
                'Macro Precision': result['macro_precision'],
                'Weighted Precision': result['weighted_precision'],
                'Macro Recall': result['macro_recall'],
                'Weighted Recall': result['weighted_recall']
            })

    df = pd.DataFrame(comparison_df)
    print(df.round(4))

    # Save comparison results
    df.to_csv('model_comparison_results.csv', index=False)

    return results


if __name__ == "__main__":
    # Create directories
    import os
    os.makedirs('models', exist_ok=True)

    # Run all models
    results = run_all_models()

    # Example of using a single model
    print(f"\n{'='*50}")
    print("SINGLE MODEL EXAMPLE - LOGISTIC REGRESSION")
    print(f"{'='*50}")

    # Initialize and train a single model
    clf = TFIDFClassifier(model_type='logistic')
    train_texts, train_labels, val_texts, val_labels = clf.prepare_data()
    clf.fit(train_texts, train_labels)
    result = clf.evaluate(val_texts, val_labels)

    # Example prediction
    sample_texts = [
        "Tôi rất thích sản phẩm này",
        "Sản phẩm này tệ quá"
    ]
    predictions, probabilities = clf.predict(sample_texts)

    print(f"\nSample Predictions:")
    for i, text in enumerate(sample_texts):
        print(f"Text: {text}")
        print(f"Prediction: {predictions[i]}")
        print(f"Probabilities: {probabilities[i]}")
        print("-" * 30)
