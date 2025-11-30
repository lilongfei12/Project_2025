import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


# Quantile Regression Model
class QuantileRegressionModel(nn.Module):

    def __init__(self, input_size=8, output_size=1):
        super(QuantileRegressionModel, self).__init__()
        # Define weights and bias, corresponding to W and b in TensorFlow
        self.w = nn.Parameter(torch.randn(input_size, output_size) * 0.01)
        self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b


# Quantile Loss Function
class QuantileLoss(nn.Module):

    def __init__(self, gama=0.75):
        super(QuantileLoss, self).__init__()
        self.gama = gama
        self.beta = 1 - gama

    # Calculate quantile loss
    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        loss = torch.where(diff > 0, diff * self.gama, -diff * self.beta)
        return torch.sum(loss)

# Diabetes Quantile Regression Classifier
class DiabetesQuantileRegression:


    def __init__(self, gama=0.75, learning_rate=0.001, model_name="QR Model"):
        self.gama = gama
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.model = QuantileRegressionModel()
        self.criterion = QuantileLoss(gama)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.loss_list = []
        self.train_accuracies = []
        self.test_accuracies = []

    def load_and_preprocess_data(self, file_path):

        # Read data
        df = pd.read_csv(file_path, encoding='gbk', header=0)

        print("Data types:")
        print(df.dtypes)
        print("\nFirst few rows of data:")
        print(df.head())

        # Convert to float32
        df = df.astype(np.float32)
        df_array = np.array(df)

        # Data normalization
        for i in range(8):
            max_val = df_array[:, i].max()
            min_val = df_array[:, i].min()
            if max_val - min_val > 0:
                df_array[:, i] = (df_array[:, i] - min_val) / (max_val - min_val)

        x_data = df_array[:, 0:8]
        y_data = df_array[:, -1]

        print(f"\nFeature data shape: {x_data.shape}")
        print(f"Label data shape: {y_data.shape}")

        # Output class distribution
        unique, counts = np.unique(y_data, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")

        return x_data, y_data

    def prepare_data(self, x_data, y_data, test_size=0.3, random_state=1221):

        # Data split
        X_train, X_test, Y_train, Y_test = train_test_split(
            x_data, y_data, test_size=test_size, random_state=random_state
        )

        # Output data shape information
        print(f"\nData split results:")
        print(f"X_train.shape = {X_train.shape}")
        print(f"Y_train.shape = {Y_train.shape}")
        print(f"X_test.shape = {X_test.shape}")
        print(f"Y_test.shape = {Y_test.shape}")

        # Output test set class distribution
        unique, counts = np.unique(Y_test, return_counts=True)
        print(f"Test set class distribution: {dict(zip(unique, counts))}")

        return X_train, X_test, Y_train, Y_test

    def calculate_accuracy(self, X, Y):
        """
        Calculate accuracy
        """
        self.model.eval()
        correct_count = 0
        total_count = len(Y)

        with torch.no_grad():
            for i in range(total_count):
                x_tensor = torch.FloatTensor(X[i].reshape(1, -1))
                prediction = self.model(x_tensor)
                predict_class = 1 if prediction.item() >= 0.5 else 0
                target_class = Y[i]
                if target_class == predict_class:
                    correct_count += 1

        return correct_count / total_count

    def train(self, X_train, Y_train, X_test, Y_test, epochs=1000):

        print(f"\nStarting training for {self.model_name}, total {epochs} epochs...")
        print(f"Quantile parameter γ = {self.gama}")

        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0

            # Shuffle data order
            X_shuffled, Y_shuffled = shuffle(X_train, Y_train)

            # Train sample by sample
            for xs, ys in zip(X_shuffled, Y_shuffled):
                xs_tensor = torch.FloatTensor(xs.reshape(1, -1))
                ys_tensor = torch.FloatTensor([ys]).view(1, 1)

                # Forward propagation
                predictions = self.model(xs_tensor)
                loss = self.criterion(predictions, ys_tensor)

                # Backward propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            # Record loss
            average_loss = total_loss
            self.loss_list.append(average_loss)

            # Calculate and record accuracy every 100 epochs
            if epoch % 100 == 0:
                train_acc = self.calculate_accuracy(X_train, Y_train)
                test_acc = self.calculate_accuracy(X_test, Y_test)
                self.train_accuracies.append(train_acc)
                self.test_accuracies.append(test_acc)

                w_temp = self.model.w.detach().numpy()
                b_temp = self.model.b.detach().numpy()
                print(f"epoch={epoch}, loss={average_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

        # Final accuracy
        final_train_acc = self.calculate_accuracy(X_train, Y_train)
        final_test_acc = self.calculate_accuracy(X_test, Y_test)
        self.train_accuracies.append(final_train_acc)
        self.test_accuracies.append(final_test_acc)

        print(f"Training completed! Final training accuracy: {final_train_acc:.4f}, Test accuracy: {final_test_acc:.4f}")

    def evaluate(self, X_test, Y_test):

        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for i in range(len(Y_test)):
                x_test_tensor = torch.FloatTensor(X_test[i].reshape(1, -1))
                prediction = self.model(x_test_tensor)
                predict_class = 1 if prediction.item() >= 0.5 else 0
                predictions.append(predict_class)
                true_labels.append(Y_test[i])

        accuracy = accuracy_score(true_labels, predictions)

        print(f"\n{self.model_name} Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")

        # Fix: Add zero_division parameter and English labels
        print(classification_report(true_labels, predictions,
                                    zero_division=0,  # Set undefined metrics to 0
                                    target_names=['Non-Diabetic', 'Diabetic']))

        # Add prediction distribution information
        unique, counts = np.unique(predictions, return_counts=True)
        pred_dist = dict(zip(unique, counts))
        print(f"Prediction distribution: Non-Diabetic={pred_dist.get(0, 0)}, Diabetic={pred_dist.get(1, 0)}")

        return accuracy, predictions, true_labels

    def plot_loss(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.model_name} - Loss Change (γ={self.gama})")
        plt.grid(True)
        plt.show()

    def plot_accuracy(self):

        if len(self.train_accuracies) > 0:
            epochs = np.linspace(0, len(self.loss_list), len(self.train_accuracies))
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, self.train_accuracies, label='Training Accuracy', marker='o')
            plt.plot(epochs, self.test_accuracies, label='Test Accuracy', marker='s')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{self.model_name} - Accuracy Change (γ={self.gama})")
            plt.legend()
            plt.grid(True)
            plt.show()

    def get_model_parameters(self):

        return {
            'w': self.model.w.detach().numpy(),
            'b': self.model.b.detach().numpy()
        }


def compare_quantile_models():

    # Define quantile parameters to compare
    quantiles = [0.125, 0.375, 0.75, 0.875]
    models = {}
    results = {}

    # Load data (all models use the same data)
    base_model = DiabetesQuantileRegression(gama=0.75)
    x_data, y_data = base_model.load_and_preprocess_data('data/diabetes.csv')
    X_train, X_test, Y_train, Y_test = base_model.prepare_data(x_data, y_data)

    print("\n" + "=" * 60)
    print("Starting comparison of models with different quantile parameters")
    print("=" * 60)

    # Train and evaluate each quantile model
    for gama in quantiles:
        model_name = f"QR_γ={gama}"
        print(f"\n{'=' * 40}")
        print(f"Training model: {model_name}")
        print(f"{'=' * 40}")

        # Create and train model
        model = DiabetesQuantileRegression(gama=gama, learning_rate=0.001, model_name=model_name)
        model.train(X_train, Y_train, X_test, Y_test, epochs=1000)

        # Evaluate model
        accuracy, predictions, true_labels = model.evaluate(X_test, Y_test)

        # Save results
        models[gama] = model
        results[gama] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'final_loss': model.loss_list[-1] if model.loss_list else 0
        }

    # Compare results
    print("\n" + "=" * 60)
    print("Model Comparison Results")
    print("=" * 60)

    comparison_df = pd.DataFrame({
        'Quantile γ': quantiles,
        'Test Accuracy': [results[gama]['accuracy'] for gama in quantiles],
        'Final Loss': [results[gama]['final_loss'] for gama in quantiles]
    })

    print(comparison_df.round(4))

    # Plot comparison graphs
    plot_comparison_results(models, results, quantiles)

    return models, results


def plot_comparison_results(models, results, quantiles):

    # 1. Accuracy comparison
    plt.figure(figsize=(12, 8))

    # Accuracy bar chart
    plt.subplot(2, 2, 1)
    accuracies = [results[gama]['accuracy'] for gama in quantiles]
    bars = plt.bar(range(len(quantiles)), accuracies, color='skyblue', alpha=0.7)
    plt.xlabel('Quantile γ')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Comparison for Different Quantile Parameters')
    plt.xticks(range(len(quantiles)), [f'{gama}' for gama in quantiles])

    # Add values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    # Loss curve comparison
    plt.subplot(2, 2, 2)
    for gama in quantiles:
        plt.plot(models[gama].loss_list, label=f'γ={gama}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve Comparison')
    plt.legend()
    plt.grid(True)

    # Training accuracy change
    plt.subplot(2, 2, 3)
    for gama in quantiles:
        epochs = np.linspace(0, len(models[gama].loss_list), len(models[gama].train_accuracies))
        plt.plot(epochs, models[gama].train_accuracies, label=f'γ={gama}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Change Comparison')
    plt.legend()
    plt.grid(True)

    # Test accuracy change
    plt.subplot(2, 2, 4)
    for gama in quantiles:
        epochs = np.linspace(0, len(models[gama].loss_list), len(models[gama].test_accuracies))
        plt.plot(epochs, models[gama].test_accuracies, label=f'γ={gama}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Change Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 2. Confusion matrix comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, gama in enumerate(quantiles):
        if idx < len(axes):
            cm = confusion_matrix(results[gama]['true_labels'], results[gama]['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Non-Diabetic', 'Diabetic'],
                        yticklabels=['Non-Diabetic', 'Diabetic'])
            axes[idx].set_title(f'γ={gama} - Accuracy: {results[gama]["accuracy"]:.3f}')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')

    # Hide extra subplots
    for idx in range(len(quantiles), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def main():

    # Compare models with different quantile parameters
    models, results = compare_quantile_models()

    # Find the best model
    best_gama = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_gama]['accuracy']

    print(f"\n{'=' * 50}")
    print(f"Best Model: γ = {best_gama}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"{'=' * 50}")

    # Show detailed information of the best model
    best_model = models[best_gama]
    best_model.plot_loss()
    best_model.plot_accuracy()

    return models, results


if __name__ == "__main__":
    models, results = main()
