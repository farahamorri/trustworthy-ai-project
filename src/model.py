import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# 1. Définition de l'architecture du Réseau de Neurones
class CreditModel(nn.Module):
    def __init__(self, input_dim):
        super(CreditModel, self).__init__()
        # Couche 1 : Prend les 23 variables et les transforme en 64 neurones
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU() # Fonction d'activation (ajoute de la non-linéarité)
        
        # Couche 2 : 64 neurones vers 32 neurones
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
        # Couche de sortie : 32 neurones vers 1 seule valeur (0 ou 1)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output_layer(x)
        return x

# 2. Fonction d'entraînement
def train_model(X_train, y_train, epochs=50, lr=0.01):
    print("\n🚀 Début de l'entraînement du modèle PyTorch...")
    
    # PyTorch ne comprend pas Pandas. Il faut convertir les données en "Tenseurs"
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    # On ajoute unsqueeze(1) pour que la forme soit [24000, 1] et non [24000]
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) 

    # Initialiser le modèle
    input_dim = X_train.shape[1] # C'est le chiffre 23
    model = CreditModel(input_dim)
    
    # Fonction de perte (Binary Cross Entropy) et Optimiseur (Adam)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Boucle d'apprentissage
    for epoch in range(epochs):
        model.train() # Met le modèle en mode entraînement
        optimizer.zero_grad() # Remet les gradients à zéro
        
        outputs = model(X_tensor) # Le modèle fait ses prédictions
        loss = criterion(outputs, y_tensor) # On calcule l'erreur
        
        loss.backward() # Rétropropagation (calcul des gradients)
        optimizer.step() # Mise à jour des poids

        # Afficher la progression toutes les 10 époques
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Error (Loss): {loss.item():.4f}")

    return model

# 3. Fonction d'évaluation (pour voir s'il est performant)
def evaluate_model(model, X_test, y_test):
    model.eval() # Met le modèle en mode test (bloque l'apprentissage)
    
    X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    
    with torch.no_grad(): # On ne calcule pas les gradients pour gagner du temps/mémoire
        outputs = model(X_tensor)
        # On passe la sortie dans une Sigmoïde pour avoir une probabilité entre 0 et 1
        # Puis on arrondit (round) pour obtenir la classe 0 ou 1
        predictions = torch.sigmoid(outputs).round()

    # Calcul de la précision
    acc = accuracy_score(y_test, predictions.numpy())
    print(f"\n🎯 Précision (Accuracy) sur le set de test : {acc * 100:.2f}%\n")
    return acc