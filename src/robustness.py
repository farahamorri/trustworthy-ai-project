import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import LowProFool
from sklearn.metrics import accuracy_score


from src.model import CreditModel

def attack_fgsm(model, X_test, y_test, epsilon=0.2):
    """Generates an FGSM attack on the test set and evaluates the model's vulnerability."""
    print(f"\n⚔️ Phase 3: Starting FGSM Adversarial Attack (by Farah) with epsilon={epsilon}...")
    
    # 1. Preparation for ART (Adversarial Robustness Toolbox)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # ART needs the bounds of our data
    min_val = float(X_test.min().min())
    max_val = float(X_test.max().max())
    
    # Wrap our PyTorch model so the ART library can interact with it
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_val, max_val),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(X_test.shape[1],),
        nb_classes=2 # Binary classification
    )
    
    # 2. Attack Configuration
    # "eps" (epsilon) is the strength of the perturbation.
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    
    # 3. Generating corrupted/malicious data
    print(f"Generating adversarial examples (Perturbation epsilon = {epsilon})...")
    X_test_numpy = X_test.values.astype(np.float32)
    
    # The attack modifies X_test_numpy to create X_test_adv
    X_test_adv = attack.generate(x=X_test_numpy)
    
    # 4. Evaluating the model on this spoofed data
    X_test_adv_tensor = torch.tensor(X_test_adv, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        outputs_adv = model(X_test_adv_tensor)
        predictions_adv = torch.sigmoid(outputs_adv).round().numpy().flatten()
        
    acc_adv = accuracy_score(y_test, predictions_adv)
    
    print(f"🛑 Accuracy under FGSM attack: {acc_adv * 100:.2f}%")
    
    # We return the corrupted test set so we can test Olivia's defense shield later!
    return X_test_adv_tensor


def train_robust_model(X_train, y_train, baseline_model, class_weights=None, epsilon=0.2, epochs=300, lr=0.01, dropout_p=0.4):
    """Defends the model via Adversarial Training (training on both clean and attacked data)."""
    print("\n🛡️ Phase 3: Defense via Adversarial Training (by Olivia)...")
    
    # 1. Use the baseline model to generate attacks on the TRAINING set
    classifier = PyTorchClassifier(
        model=baseline_model,
        clip_values=(float(X_train.min().min()), float(X_train.max().max())),
        loss=nn.BCEWithLogitsLoss(),
        optimizer=optim.Adam(baseline_model.parameters(), lr=lr),
        input_shape=(X_train.shape[1],),
        nb_classes=2
    )
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    
    print("Generating poison on training data (this may take a few seconds)...")
    X_train_numpy = X_train.values.astype(np.float32)
    X_train_adv = attack.generate(x=X_train_numpy)
    
    # 2. Mix clean data (24,000) and attacked data (24,000)
    # The model will train on 48,000 examples!
    X_combined = np.vstack((X_train_numpy, X_train_adv))
    y_combined = np.vstack((y_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1)))
    
    X_tensor = torch.tensor(X_combined, dtype=torch.float32)
    y_tensor = torch.tensor(y_combined, dtype=torch.float32)
    
    # 3. Train a BRAND NEW model (The Shield)
    print("Training the robust shield model...")
    robust_model = CreditModel(X_train.shape[1], dropout_p=dropout_p)
    
    # 🔴 CRITICAL FIX: Add Class Weights to maintain business value (Recall)
    if class_weights is not None:
        pos_weight_value = class_weights[1] / class_weights[0]
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.Adam(robust_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        robust_model.train()
        optimizer.zero_grad()
        outputs = robust_model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print("✅ Robust training complete.")
    return robust_model


def attack_fgsm_smart(model, X_test, y_test, X_test_columns):
    print("\n🥷 Phase 3: Starting Smart FGSM Attack (by Farah)...")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(float(X_test.min().min()), float(X_test.max().max())),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(X_test.shape[1],),
        nb_classes=2
    )
    
    # 1. CREATION OF CUSTOM EPSILON VECTORS
    eps_array = np.zeros(X_test.shape[1])
    
    # 🔴 THE FIX: Create a separate step array that is strictly > 0 everywhere
    # It must be the same shape and type as eps_array to satisfy ART's strict type checking
    eps_step_array = np.full(X_test.shape[1], 0.1)
    
    for i, col_name in enumerate(X_test_columns):
        if col_name in ['SEX', 'EDUCATION', 'MARRIAGE'] or col_name.startswith('PAY_'):
            # Max perturbation allowed is 0.0 (Data is frozen)
            eps_array[i] = 0.0
        else:
            # Max perturbation allowed is 0.1
            eps_array[i] = 0.1 

    print("Generated max perturbation vector (Epsilons):", eps_array)
    
    # 2. ATTACK WITH THE VECTORS
    attack = FastGradientMethod(
        estimator=classifier, 
        eps=eps_array, 
        eps_step=eps_step_array # 🔴 Using our strictly positive dummy step
    )
    
    print("Generating stealthy adversarial examples...")
    X_test_numpy = X_test.values.astype(np.float32)
    X_test_adv = attack.generate(x=X_test_numpy)
    
    # 3. EVALUATION
    X_test_adv_tensor = torch.tensor(X_test_adv, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        outputs_adv = model(X_test_adv_tensor)
        predictions_adv = torch.sigmoid(outputs_adv).round().numpy().flatten()
        
    acc_adv = accuracy_score(y_test, predictions_adv)
    
    print(f"🛑 Accuracy under Smart FGSM attack: {acc_adv * 100:.2f}%\n")
    
    return X_test_adv_tensor


class ARTModelWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        
    def forward(self, x):
        # On récupère le score brut du modèle [N, 1]
        logits = self.model(x)
        # On le transforme en probabilité (entre 0 et 1)
        prob_1 = torch.sigmoid(logits)
        # La proba de la classe 0 est simplement (1 - proba de la classe 1)
        prob_0 = 1.0 - prob_1
        # On renvoie un tableau à deux colonnes [N, 2] !
        return torch.cat([prob_0, prob_1], dim=1)


def attack_lowprofool(model, X_train, y_train, X_test, y_test):
    print("\n🥷 Phase 3: Advanced Stealth Attack (LowProFool) by Farah...")
    
    # 2. On "emballe" notre modèle dans la coquille
    art_model = ARTModelWrapper(model)
    
    # On utilise MSELoss comme les chercheurs du paper (idéal pour des probabilités 2D)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(art_model.parameters(), lr=0.01)
    
    classifier = PyTorchClassifier(
        model=art_model,
        clip_values=(float(X_test.min().min()), float(X_test.max().max())),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(X_test.shape[1],),
        nb_classes=2
    )
    
    lpf_attack = LowProFool(
        classifier=classifier,
        n_steps=50,
        eta=5.0,
        lambd=1.0,
        eta_decay=0.95,
        verbose=True
    )
    
    print("Analyzing training data structure to find blind spots...")
    lpf_attack.fit_importances(X_train, y_train)
    
    # 3. CRÉATION DES CIBLES (au format 2D pour ART)
    # Si le client est un bon payeur (0), le pirate cible la classe (1)
    inverted_labels = y_test.apply(lambda x: 1 if x == 0 else 0)
    # np.eye(2) transforme [0, 1] en matrice One-Hot [[1,0], [0,1]]
    targets = np.eye(2)[inverted_labels.values]
    
    print("Generating undetectable mutants...")
    X_test_numpy = X_test.values.astype(np.float32)
    X_test_adv = lpf_attack.generate(x=X_test_numpy, y=targets)
    
    # 4. ÉVALUATION (On utilise le vrai modèle pour voir s'il s'est fait avoir)
    X_test_adv_tensor = torch.tensor(X_test_adv, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        outputs_adv = model(X_test_adv_tensor)
        predictions_adv = torch.sigmoid(outputs_adv).round().numpy().flatten()
        
    acc_adv = accuracy_score(y_test, predictions_adv)
    print(f"🛑 Global Accuracy under LowProFool attack: {acc_adv * 100:.2f}%\n")
    
    return X_test_adv_tensor


Python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import LowProFool

def train_robust_model_lpf(X_train, y_train, baseline_model, class_weights=None, epochs=300, lr=0.01):
    print("\n🛡️ Phase 3: Creation of the Ultimate Shield (Adversarial Training LPF) by Olivia...")
    
    # 1. Use the previously created wrapper so PyTorch and ART can communicate
    art_model = ARTModelWrapper(baseline_model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(art_model.parameters(), lr=0.01)
    
    classifier = PyTorchClassifier(
        model=art_model,
        clip_values=(float(X_train.min().min()), float(X_train.max().max())),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(X_train.shape[1],),
        nb_classes=2
    )
    
    # 2. LowProFool configuration (accelerated version for training)
    lpf_attack = LowProFool(
        classifier=classifier,
        n_steps=50, 
        eta=5.0,
        lambd=1.0,
        eta_decay=0.95,
        verbose=True
    )
    
    print("Analyzing the structure of the training data...")
    lpf_attack.fit_importances(X_train, y_train)
    
    # 3. Creation of inverted targets for the TRAINING set
    inverted_labels_train = y_train.apply(lambda x: 1 if x == 0 else 0)
    targets_train = np.eye(2)[inverted_labels_train.values]
    
    print("⚠️ GENERATING THE VACCINE ON 24,000 ROWS... (Grab a coffee, this will take a while) ☕")
    X_train_numpy = X_train.values.astype(np.float32)
    X_train_adv = lpf_attack.generate(x=X_train_numpy, y=targets_train)
    
    # 4. Mixing: Combine the 24,000 real clients and the 24,000 mutants (48,000 total)
    X_combined = np.vstack((X_train_numpy, X_train_adv))
    y_combined = np.vstack((y_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1)))
    
    X_tensor = torch.tensor(X_combined, dtype=torch.float32)
    y_tensor = torch.tensor(y_combined, dtype=torch.float32)
    
    # 5. Training the Robust model
    print("Training the robust shield model...")
    # Initialize a brand new network (ensure CreditModel is properly imported)
    robust_model = CreditModel(X_train.shape[1], dropout_p=0.4)
    
    # 🔴 Reapply class weights to maintain our business performance (Recall)
    if class_weights is not None:
        pos_weight_value = class_weights[1] / class_weights[0]
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        
    opt = optim.Adam(robust_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        robust_model.train()
        opt.zero_grad()
        outputs = robust_model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        loss.backward()
        opt.step()
        scheduler.step()

    print("✅ LowProFool shield activated and model trained.")
    return robust_model