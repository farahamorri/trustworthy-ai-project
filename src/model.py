import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

# 1. Neural Network Architecture Definition
class CreditModel(nn.Module):
    def __init__(self, input_dim, dropout_p=0.3):
        super(CreditModel, self).__init__()
        # Layer 1: Input to 64 neurons
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        # Dropout layer after first activation
        # Randomly zeroes out neurons during training to prevent overfitting
        self.dropout1 = nn.Dropout(p=dropout_p)
        
        # Layer 2: 64 neurons to 32 neurons
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        # Dropout layer after second activation
        self.dropout2 = nn.Dropout(p=dropout_p)
        
        # Output layer: 32 neurons to 1 single value
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        # Apply dropout only during training (handled by model.train()/eval())
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.output_layer(x) 
        return x

# 2. Training Function
def train_model(X_train, y_train, epochs=100, start_lr=0.01, class_weights=None, dropout_p=0.3, instance_weights=None):
    print("\n🚀 Starting PyTorch model training with Regularization...")
    print(f"Initial LR: {start_lr}, Epochs: {epochs}, Dropout: {dropout_p}")
    
    # PyTorch doesn't understand Pandas. We must convert data to "Tensors"
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    # We add unsqueeze(1) so the shape is [N, 1] and not [N]
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) 

    # Initialize the model
    input_dim = X_train.shape[1] 
    model = CreditModel(input_dim, dropout_p=dropout_p)

    if instance_weights is not None:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    if class_weights is not None:
        # pos_weight is the ratio of negative to positive samples (Weight of Class 1 / Weight of Class 0)
        pos_weight_value = class_weights[1] / class_weights[0]
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss() 
        
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    # Learning Rate Scheduler (Cosine Annealing)
    # T_max is the number of epochs to reach the minimum LR (eta_min).
    # It smoothly decays LR from start_lr down to ~0 over the total epochs.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Training loop
    for epoch in range(epochs):
        model.train() # Set the model to training mode
        optimizer.zero_grad() # Reset gradients to zero
        
        outputs = model(X_tensor) # Model makes predictions
        loss = criterion(outputs, y_tensor) # Calculate the error

        if instance_weights is not None:
            loss = (loss * instance_weights).mean()
        else:
            loss = loss.mean()
        
        loss.backward() # Backpropagation (calculate gradients)
        optimizer.step() # Update weights
        scheduler.step()

        # Display progress and current LR every 20 epochs
        if (epoch+1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Current LR: {current_lr:.6f}")

    return model

# 3. Evaluation Function
def evaluate_model(model, X_test, y_test):
    model.eval() # Set model to evaluation mode (disables dropout/batchnorm if any)
    
    X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    
    with torch.no_grad(): # Disable gradient calculation to save time/memory
        outputs = model(X_tensor)
        # Pass output through Sigmoid to get probabilities between 0 and 1
        # Then round to get class 0 or 1
        predictions = torch.sigmoid(outputs).round()

    # Calculate metrics
    pred_numpy = predictions.numpy()
    acc = accuracy_score(y_test, pred_numpy)
    print(f"\n🎯 Accuracy on test set: {acc * 100:.2f}%\n")
    
    # Bonus: Detailed report to see performance on Class 0 vs Class 1
    print("--- Classification Report ---")
    print(classification_report(y_test, pred_numpy))
    
    return acc