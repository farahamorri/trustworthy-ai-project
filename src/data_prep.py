import pandas as pd
import os
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():

    # Désactive la vérification SSL pour permettre le téléchargement
    ssl._create_default_https_context = ssl._create_unverified_context
    # Créer le dossier data s'il n'existe pas sur l'ordinateur de Farah ou Olivia
    os.makedirs("data/raw", exist_ok=True)
    file_path = "data/raw/default_of_credit_card_clients.csv"
    
    # Si le fichier n'a pas encore été téléchargé, on le télécharge
    if not os.path.exists(file_path):
        print("Téléchargement du dataset depuis UCI...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        
        # Pandas peut lire directement depuis une URL et on sauvegarde en local
        df = pd.read_excel(url, header=1)
        df.to_csv(file_path, index=False)
        print("✅ Dataset téléchargé et converti en CSV dans data/raw/")
    else:
        print("✅ Dataset CSV déjà présent en local.")
        df = pd.read_csv(file_path)
        
    return df


def prepare_data(df):
    print("Début du pré-traitement des données...")
    
    # 1. Supprimer la colonne ID qui ne sert à rien
    df = df.drop(columns=['ID'])
    
    # 2. Renommer la variable cible pour que ce soit plus simple
    df = df.rename(columns={'default payment next month': 'default'})
    
    # 3. Séparer les caractéristiques (X) de la cible (y)
    X = df.drop(columns=['default'])
    y = df['default']
    
    # 4. Séparation en données d'entraînement (80%) et de test (20%)
    # On met un random_state=42 pour que Farah et Olivia aient exactement la même séparation !
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Standardisation (Mise à l'échelle)
    scaler = StandardScaler()
    
    # On "apprend" l'échelle uniquement sur le Train, et on l'applique au Train et au Test
    # (Pour éviter de tricher en regardant les données de Test)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    
    print("✅ Données nettoyées, standardisées et séparées !")
    print(f"Taille du set d'entraînement : {X_train_scaled.shape}")
    print(f"Taille du set de test : {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test