import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader # 🔴 L'import qui manquait !
from opacus import PrivacyEngine
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from sklearn.metrics import accuracy_score
from src.model import CreditModel

def attack_mia(model, X_train, y_train, X_test, y_test):
    print("\n🕵️‍♀️ Phase 4 : Attaque MIA (Rule-Based) par Olivia...")
    
    model.eval()
    with torch.no_grad():
        # Le pirate interroge le modèle sur les données Train
        out_train = model(torch.tensor(X_train.values, dtype=torch.float32))
        pred_train = torch.sigmoid(out_train).round().numpy().flatten()
        
        # Le pirate interroge le modèle sur les données Test
        out_test = model(torch.tensor(X_test.values, dtype=torch.float32))
        pred_test = torch.sigmoid(out_test).round().numpy().flatten()

    # Le pirate regarde si le modèle a eu juste
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)
    
    # Formule mathématique du succès de l'attaque
    # Le pirate a 50% de chance de base (hasard). Il gagne des points si le modèle
    # est beaucoup plus performant sur le Train que sur le Test (Overfitting).
    mia_success_rate = 50.0 + ((acc_train - acc_test) / 2) * 100
    
    print(f"🛑 Taux de succès de l'attaque MIA : {mia_success_rate:.2f}%")
    if mia_success_rate > 55.0:
        print("⚠️ ALERTE : Le modèle fuit des informations privées (Overfitting) !")
    else:
        print("✅ Le modèle protège bien sa vie privée (le pirate tire au hasard).")
        
    return mia_success_rate


def train_private_model(X_train, y_train, epochs=15, lr=0.01):
    print("\n🛡️ Phase 4 : Défense par Differential Privacy (DP-SGD) par Farah...")
    
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=256)
    
    model = CreditModel(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    privacy_engine = PrivacyEngine()
    
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0, 
        max_grad_norm=1.0,
    )
    
    print("Entraînement du modèle privé (ajout de bruit mathématique)...")
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    epsilon = privacy_engine.get_epsilon(1e-5)
    print(f"🔒 Entraînement terminé ! Budget de confidentialité (Epsilon) : {epsilon:.2f}")
    
    return model