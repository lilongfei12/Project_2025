import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DiabetesClassifier:

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self, file_path):

        # Load dataset
        df = pd.read_csv(file_path)
        print("Dataset Overview:")
        print(df.head())

        # Dataset information
        print(f"\nDataset Shape: {df.shape}")
        print(f"Features: {df.columns[:-1].tolist()}")
        print(f"Target variable: {df.columns[-1]}")

        '''
        Feature Description:
        - Pregnancies: Number of pregnancies
        - Glucose: Plasma glucose concentration
        - BloodPressure: Diastolic blood pressure (mm Hg)
        - SkinThickness: Triceps skin fold thickness (mm)
        - Insulin: 2-Hour serum insulin (mu U/ml)
        - BMI: Body mass index (weight in kg/(height in m)^2)
        - DiabetesPedigreeFunction: Diabetes pedigree function
        - Age: Age in years
        - Outcome: Class variable (0: non-diabetic, 1: diabetic)

        Among 768 samples: 500 labeled as 0 (non-diabetic), 268 as 1 (diabetic)
        '''

        # Split features and target
        X = df[df.columns[0:-1]].values  # First eight columns as features
        Y = df[df.columns[-1]].values  # Last column as target (0: healthy, 1: diabetic)

        return X, Y

    def prepare_data(self, X, Y, test_size=0.3, random_state=1221):

        # Split data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )

        # Print dataset shapes
        print(f"\nData Shapes:")
        print(f"X_train.shape = {X_train.shape}")
        print(f"Y_train.shape = {Y_train.shape}")
        print(f"X_test.shape = {X_test.shape}")
        print(f"Y_test.shape = {Y_test.shape}")

        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train = torch.from_numpy(X_train.astype(np.float32))
        Y_train = torch.from_numpy(Y_train.astype(np.float32))
        X_test = torch.from_numpy(X_test.astype(np.float32))
        Y_test = torch.from_numpy(Y_test.astype(np.float32))

        # Reshape target tensors to 2D
        Y_train = Y_train.view(Y_train.shape[0], 1)
        Y_test = Y_test.view(Y_test.shape[0], 1)

        print(f"Target shapes after reshaping: {Y_train.size()}, {Y_test.size()}")

        return X_train, X_test, Y_train, Y_test

#Neural network model for binary classification

class ClassificationModel(torch.nn.Module):

    def __init__(self, in_features):

        super(ClassificationModel, self).__init__()
        # Define neural network layers
        self.linear = torch.nn.Linear(in_features, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        pred = self.linear(x)
        out = self.sigmoid(pred)  # Convert to probabilities [0,1]
        return out


class ModelTrainer:

    def __init__(self, model, learning_rate=0.001):

        self.model = model
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def train(self, X_train, Y_train, epochs=100000):

        print(f"\nStarting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.model(X_train)

            # Compute loss
            loss = self.criterion(y_pred, Y_train)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Print progress every 10000 epochs
            if (epoch + 1) % 10000 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        print(f"Training completed! Final loss: {loss.item():.4f}")

    def evaluate(self, X_test, Y_test):

        with torch.no_grad():
            # Make predictions
            y_pred = self.model(X_test)
            y_pred_cls = y_pred.round()

            # Calculate accuracy
            correct_predictions = y_pred_cls.eq(Y_test).sum().item()
            total_samples = Y_test.shape[0]
            accuracy = correct_predictions / total_samples

            '''
            Explanation:
            - y_pred_cls: Predicted classes (0 or 1) for test set
            - y_pred_cls.eq(Y_test).sum(): Count of correct predictions
            - total_samples: Total number of test samples
            - accuracy = correct_predictions / total_samples
            '''

            print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
            return accuracy


def main():

    # Initialize data processor
    processor = DiabetesClassifier()

    # Load and preprocess data
    X, Y = processor.load_and_preprocess_data('data/diabetes.csv')

    # Prepare data for training
    X_train, X_test, Y_train, Y_test = processor.prepare_data(X, Y)

    # Initialize model
    n_features = X.shape[1]
    model = ClassificationModel(n_features)

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nModel Parameters: {len(list(model.parameters()))} parameter groups")

    # Initialize trainer
    trainer = ModelTrainer(model, learning_rate=0.001)

    # Train model
    trainer.train(X_train, Y_train, epochs=100000)

    # Evaluate model
    accuracy = trainer.evaluate(X_test, Y_test)

    return model, accuracy


if __name__ == "__main__":
    model, accuracy = main()