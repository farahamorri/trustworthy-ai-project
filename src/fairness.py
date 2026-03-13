import torch
import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate

def evaluate_fairness(model, X_test, y_test):
    print("\n⚖️ Début de l'audit d'équité (Fairness) par Farah...")
    
    # 1. Générer les prédictions du modèle
    model.eval()
    X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        # On récupère les prédictions sous forme de tableau 1D (0 ou 1)
        predictions = torch.sigmoid(outputs).round().numpy().flatten()
        
    y_true = y_test.values

    # 2. Reconstruire l'attribut sensible (Le Sexe)
    # Grâce au StandardScaler, Sexe = 1 (Homme) est devenu < 0, et Sexe = 2 (Femme) est devenu > 0.
    sensitive_feature = (X_test['SEX'] > 0).astype(int) 
    # Mappage : 0 = Hommes, 1 = Femmes
    
    # 3. Calculer les métriques avec Fairlearn
    # "selection_rate" mesure le % de fois où le modèle prédit "1" (Risque de défaut)
    mf = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=predictions,
        sensitive_features=sensitive_feature
    )
    
    taux_hommes = mf.by_group[0] * 100
    taux_femmes = mf.by_group[1] * 100
    
    print(f"Taux de prédiction de 'Défaut' pour les Hommes : {taux_hommes:.2f}%")
    print(f"Taux de prédiction de 'Défaut' pour les Femmes : {taux_femmes:.2f}%")
    
    # Différence de parité (L'écart absolu entre les deux groupes)
    dpd = demographic_parity_difference(y_true, predictions, sensitive_features=sensitive_feature)
    print(f"🛑 Différence de parité démographique : {dpd * 100:.2f}%")
    
    if dpd > 0.05:
        print("⚠️ ALERTE : Le modèle est biaisé. L'écart entre les genres dépasse 5%.")
    else:
        print("✅ Le modèle est considéré comme équitable (écart < 5%).")
        
    return mf



def get_sample_weights(X_train, y_train):
    print("\n⚖️ Calcul des poids de mitigation (Kamiran & Calders) par Olivia...")
    
    sensitive_attr = (X_train['SEX'] > 0).astype(int).values
    y = y_train.values
    N = len(y)
    
    weights = np.zeros(N)
    
    for a in [0, 1]:
        for target in [0, 1]:
            # Trouver les masques
            mask_a = (sensitive_attr == a)
            mask_y = (y == target)
            mask_ay = mask_a & mask_y
            
            # Compter les effectifs
            count_a = mask_a.sum()
            count_y = mask_y.sum()
            count_ay = mask_ay.sum()
            
            # Application de la formule mathématique exacte
            if count_ay > 0:
                weights[mask_ay] = (count_a * count_y) / (N * count_ay)
                
    return torch.tensor(weights, dtype=torch.float32).unsqueeze(1)