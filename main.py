from src.data_prep import load_data, prepare_data
from src.model import train_model, evaluate_model
from src.fairness import evaluate_fairness, get_sample_weights
from src.robustness import attack_fgsm, train_robust_model
from src.privacy import attack_mia, train_private_model

if __name__ == "__main__":
    # --- PHASE 1 : DATA (Farah) ---
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # --- MODÈLE 1 : BIAISÉ (Baseline) ---
    print("\n" + "="*50)
    print("MODÈLE 1 : BASELINE (Sans mitigation)")
    print("="*50)
    baseline_model = train_model(X_train, y_train, epochs=50, lr=0.01)
    acc_base = evaluate_model(baseline_model, X_test, y_test)
    evaluate_fairness(baseline_model, X_test, y_test)
    
    # # --- MODÈLE 2 : CORRIGÉ (Reweighting) ---
    # print("\n" + "="*50)
    # print("MODÈLE 2 : CORRIGÉ (Avec Reweighting)")
    # print("="*50)
    # weights = get_sample_weights(X_train, y_train)
    # fair_model = train_model(X_train, y_train, epochs=50, lr=0.01, sample_weights=weights)
    # acc_fair = evaluate_model(fair_model, X_test, y_test)
    # evaluate_fairness(fair_model, X_test, y_test)

    # --- PHASE 3 : ROBUSTESSE (Attaque) ---
    print("\n" + "="*50)
    print("PHASE 3 : SÉCURITÉ ET ROBUSTESSE")
    print("="*50)
    
    # Farah attaque le modèle de base !
    X_test_adv_tensor = attack_fgsm(baseline_model, X_test, y_test, epsilon=1.5)

    # 2. Olivia crée le bouclier
    robust_model = train_robust_model(X_train, y_train, baseline_model, epsilon=1.5)
    
    # 3. Farah attaque le NOUVEAU modèle robuste pour voir s'il résiste mieux !
    print("\n--- Attaque sur le modèle Protégé (Bouclier) ---")
    attack_fgsm(robust_model, X_test, y_test, epsilon=1.5)


    print("\n" + "="*50)
    print("PHASE 4 : CONFIDENTIALITÉ (Privacy)")
    print("="*50)
    
    # 1. Olivia attaque le modèle sans défense
    print("\n--- Attaque MIA sur le modèle Non-Protégé ---")
    attack_mia(baseline_model, X_train, y_train, X_test, y_test)
    
    # 2. Farah entraîne le modèle ultra-sécurisé avec Opacus
    private_model = train_private_model(X_train, y_train, epochs=15)
    
    # Évaluation de la précision du modèle privé
    print("\nÉvaluation de la précision du Modèle Privé :")
    acc_private = evaluate_model(private_model, X_test, y_test)
    
    # 3. Olivia tente de ré-attaquer le modèle privé !
    print("\n--- Attaque MIA sur le modèle Protégé (DP-SGD) ---")
    
    # Opacus modifie légèrement la structure du modèle, on le remet en format standard pour l'attaque
    # via _module pour que la librairie ART puisse le lire.
    attack_mia(private_model._module, X_train, y_train, X_test, y_test)