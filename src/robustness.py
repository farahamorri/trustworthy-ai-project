import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from sklearn.metrics import accuracy_score

from src.model import CreditModel

def attack_fgsm(model, X_test, y_test, epsilon=0.2):
    print("\n⚔️ Phase 3 : Début de l'attaque adverse FGSM (par Farah)...")
    
    # 1. Préparation pour ART (Adversarial Robustness Toolbox)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # ART a besoin de connaître les valeurs minimales et maximales de nos données
    min_val = float(X_test.min().min())
    max_val = float(X_test.max().max())
    
    # On "encapsule" notre modèle PyTorch dans un format que la librairie ART comprend
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_val, max_val),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(X_test.shape[1],),
        nb_classes=2 # Classification binaire
    )
    
    # 2. Configuration de l'attaque
    # "eps" (epsilon) est la force de la perturbation. 
    # Plus il est grand, plus on modifie la donnée pour tromper le modèle.
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    
    # 3. Génération des données corrompues/malveillantes
    print(f"Génération d'exemples adverses (Perturbation epsilon = {epsilon})...")
    X_test_numpy = X_test.values.astype(np.float32)
    
    # L'attaque modifie X_test_numpy pour créer X_test_adv (adversarial)
    X_test_adv = attack.generate(x=X_test_numpy)
    
    # 4. Évaluation du modèle sur ces nouvelles données truquées
    X_test_adv_tensor = torch.tensor(X_test_adv, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        outputs_adv = model(X_test_adv_tensor)
        predictions_adv = torch.sigmoid(outputs_adv).round().numpy()
        
    acc_adv = accuracy_score(y_test, predictions_adv)
    
    print(f"🛑 Précision sous attaque FGSM : {acc_adv * 100:.2f}%")
    
    # On retourne les données corrompues, car Olivia en aura besoin pour créer la défense !
    return X_test_adv_tensor


def train_robust_model(X_train, y_train, baseline_model, epsilon=0.5, epochs=50, lr=0.01):
    print("\n🛡️ Phase 3 : Défense par Adversarial Training (par Olivia)...")
    
    # 1. On utilise le modèle de base pour générer des attaques sur le set d'ENTRAÎNEMENT
    classifier = PyTorchClassifier(
        model=baseline_model,
        clip_values=(float(X_train.min().min()), float(X_train.max().max())),
        loss=nn.BCEWithLogitsLoss(),
        optimizer=optim.Adam(baseline_model.parameters(), lr=0.01),
        input_shape=(X_train.shape[1],),
        nb_classes=2
    )
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    
    print("Génération de poison sur les données d'entraînement (ça peut prendre quelques secondes)...")
    X_train_numpy = X_train.values.astype(np.float32)
    X_train_adv = attack.generate(x=X_train_numpy)
    
    # 2. On mélange les données saines (24 000) et les données attaquées (24 000)
    # Le modèle va s'entraîner sur 48 000 exemples !
    X_combined = np.vstack((X_train_numpy, X_train_adv))
    y_combined = np.vstack((y_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1)))
    
    X_tensor = torch.tensor(X_combined, dtype=torch.float32)
    y_tensor = torch.tensor(y_combined, dtype=torch.float32)
    
    # 3. On entraîne un tout NOUVEAU modèle (Le Bouclier)
    print("Entraînement du modèle bouclier robuste...")
    robust_model = CreditModel(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(robust_model.parameters(), lr=lr)

    for epoch in range(epochs):
        robust_model.train()
        optimizer.zero_grad()
        outputs = robust_model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return robust_model