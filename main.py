from src.data_prep import load_data, prepare_data
from src.model import train_model, evaluate_model

if __name__ == "__main__":
    # --- PHASE 1 : DATA (Farah) ---
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # --- PHASE 1 : MODÈLE (Olivia) ---
    # 1. On entraîne le modèle de base (Baseline)
    baseline_model = train_model(X_train, y_train, epochs=50, lr=0.01)
    
    # 2. On évalue sa précision
    accuracy = evaluate_model(baseline_model, X_test, y_test)