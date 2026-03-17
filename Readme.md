### Phase 1 : Les Fondations (Data & Modèle)

*Ici, vous préparez le terrain. L'une gère la donnée, l'autre l'algorithme.*

* **Tâche 1 (Farah) : Préparation des données.**
* Télécharger le dataset avec `pandas`.
* Nettoyer les données (gérer les valeurs manquantes si besoin).
* Standardiser/Normaliser les montants financiers avec `scikit-learn` (`StandardScaler`).
* Séparer le dataset en ensembles d'entraînement (Train) et de test (Test).


* **Tâche 2 (Olivia) : Création du Modèle PyTorch.**
* Définir l'architecture du réseau de neurones (MLP) avec `torch.nn`.
* Écrire la boucle d'entraînement (calcul de la *loss*, *backpropagation*).
* Entraîner le modèle de base sur les données fournies par Farah et calculer la précision initiale (Accuracy/F1-score).



---

### Phase 2 : Équité et Biais (Fairness)

*L'une devient l'auditrice, l'autre la correctrice.*

* **Tâche 3 (Farah) : Audit du biais (Avant mitigation).**
* Utiliser la librairie `Fairlearn` ou `AIF360` sur les prédictions du modèle d'Olivia.
* Calculer les métriques d'équité (ex: *Disparate Impact* ou *Equal Opportunity Difference*) en ciblant l'attribut sensible `SEX` ou `AGE`.
* Générer un graphique montrant le biais initial.


* **Tâche 4 (Olivia) : Mitigation du biais (Après mitigation).**
* Implémenter la méthode de *Reweighting* (re-pondération des données d'entraînement pour équilibrer l'importance des minorités/majorités).
* Ré-entraîner le modèle PyTorch avec ces nouveaux poids.
* Fournir le nouveau modèle à Farah pour qu'elle confirme que le biais a diminué.



---

### Phase 3 : Robustesse (Attaques Adverses)

*Farah prend la casquette de "Hacker", Olivia celle de "Protectrice".*

* **Tâche 5 (Farah) : L'Attaque (FGSM).**
* Utiliser la librairie `ART` (Adversarial Robustness Toolbox) pour implémenter l'attaque Fast Gradient Sign Method.
* Générer des exemples adverses (perturber les données de test).
* Prouver que la précision du modèle s'effondre face à ces données modifiées.


* **Tâche 6 (Olivia) : La Défense (Adversarial Training).**
* Prendre le code de l'attaque de Farah et l'intégrer dans la boucle d'entraînement PyTorch.
* Entraîner un modèle robuste en lui montrant des exemples normaux ET des exemples adverses.
* Prouver que ce nouveau modèle résiste mieux à l'attaque de Farah.



---

### Phase 4 : Confidentialité (Privacy)

*On inverse les rôles ! Olivia devient la "Hacker", Farah devient la "Protectrice".*

* **Tâche 7 (Olivia) : L'Attaque d'Inférence (MIA).**
* Implémenter une *Membership Inference Attack* (toujours via la librairie `ART`).
* Essayer de deviner quelles données faisaient partie du set d'entraînement et lesquelles faisaient partie du set de test.
* Mesurer le taux de succès de l'attaque (s'il est > 50%, le modèle fuit des données).


* **Tâche 8 (Farah) : La Défense (Differential Privacy).**
* Utiliser la librairie `Opacus` de PyTorch.
* Modifier la boucle d'entraînement pour appliquer l'algorithme DP-SGD (ajout de bruit mathématique et *clipping* des gradients).
* Prouver à Olivia que son attaque MIA ne fonctionne plus (le taux de succès retombe à ~50%).



---

### Phase 5 : Synthèse et Rapport (Ensemble)

*C'est la partie cruciale pour votre note finale.*

* **Tâche 9 (Farah & Olivia) : Analyse des "Trade-offs" (Compromis).**
* Rassembler vos résultats et créer un tableau comparatif.
* Expliquer pourquoi l'équité fait baisser la précision.
* Expliquer pourquoi la *Differential Privacy* (le bruit ajouté par Farah) détériore les performances globales.


* **Tâche 10 (Farah & Olivia) : Rédaction.**
* Farah rédige les justifications techniques de la préparation des données, de l'audit des biais, de l'attaque FGSM et de la défense DP-SGD.
* Olivia rédige les justifications techniques du modèle PyTorch, de la mitigation des biais, de la défense adverse et de l'attaque MIA.



**Comment vous sentez-vous face à cette répartition ?** Si elle vous convient, voulez-vous que je vous donne le script Python de la **Tâche 1 pour Farah** ou de la **Tâche 2 pour Olivia** pour que vous puissiez commencer à coder tout de suite ?