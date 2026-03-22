import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score
# Make sure to import your model architecture
from src.model import CreditModel 

def attack_mia(model, X_train, y_train, X_test, y_test):
    """Simulates a rule-based Membership Inference Attack based on generalization gap."""
    print("\n🕵️‍♀️ Phase 4: MIA Attack (Rule-Based) by Olivia...")
    
    model.eval()
    with torch.no_grad():
        # Attacker queries the model on Train data
        out_train = model(torch.tensor(X_train.values, dtype=torch.float32))
        pred_train = torch.sigmoid(out_train).round().numpy().flatten()
        
        # Attacker queries the model on Test data
        out_test = model(torch.tensor(X_test.values, dtype=torch.float32))
        pred_test = torch.sigmoid(out_test).round().numpy().flatten()

    # Attacker checks if the model was correct
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)
    
    # Mathematical formula for attack success
    # Baseline is 50% (random guessing). The attacker gains advantage if the model
    # performs significantly better on Train than Test (Overfitting).
    mia_success_rate = 50.0 + ((acc_train - acc_test) / 2) * 100
    
    print(f"🛑 MIA Attack Success Rate: {mia_success_rate:.2f}%")
    if mia_success_rate > 55.0:
        print("⚠️ ALERT: The model leaks private information (Overfitting detected)!")
    else:
        print("✅ Privacy preserved: The model is robust against this MIA (attacker is guessing randomly).")
        
    return mia_success_rate


def train_private_model(X_train, y_train, class_weights=None, epochs=15, lr=0.01):
    """Trains a model with Differential Privacy (DP-SGD) using Opacus."""
    print("\n🛡️ Phase 4: Differential Privacy Defense (DP-SGD) by Farah...")
    
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    # Batch size is crucial in DP-SGD. 256 is a good standard.
    dataloader = DataLoader(dataset, batch_size=256) 
    
    input_dim = X_train.shape[1]
    model = CreditModel(input_dim)
    
    # 🔴 CRITICAL FIX: Re-introduce class weights so the model still detects defaults
    if class_weights is not None:
        pos_weight_value = class_weights[1] / class_weights[0]
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize Opacus Privacy Engine
    privacy_engine = PrivacyEngine()
    
    # Wrap the model, optimizer, and dataloader for privacy tracking
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0, # The amount of noise added to gradients
        max_grad_norm=1.0,    # Clipping threshold for gradients
    )
    
    print("Training private model (injecting mathematical noise)...")
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Calculate the privacy budget spent
    epsilon = privacy_engine.get_epsilon(1e-5) # 1e-5 is delta (chance of DP failure)
    print(f"🔒 Training complete! Privacy Budget Spent (Epsilon): {epsilon:.2f}")
    
    return model