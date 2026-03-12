# Trustworthy AI Project: Robustness, Privacy, and Fairness

## 📌 Project Description
This project explores the design, evaluation, and securing of a Deep Learning model (Multi-Layer Perceptron) trained on the **Default of Credit Card Clients** dataset. The goal is to analyze and address three fundamental pillars of Trustworthy AI:
1. **Fairness:** Auditing and mitigating sociodemographic biases.
2. **Robustness:** Evaluating vulnerabilities and defending against adversarial attacks (Adversarial Machine Learning).
3. **Privacy:** Protecting the model against training data leakage.

## 👥 Team and Task Allocation
To ensure both team members gain comprehensive, hands-on experience, we adopted a **"Cross-Skilling"** approach. Both of us will work on building, attacking, and defending the model across different phases.

### Phase 1: Foundations (Data & Model)
* **Farah - Data Preparation:** Downloading the dataset using `pandas`, cleaning, standardizing numerical features with `scikit-learn` (`StandardScaler`), and creating the Train/Test splits.
* **Olivia - PyTorch Model:** Defining the MLP architecture using `torch.nn`, writing the training loop (loss calculation, backpropagation), and establishing the baseline accuracy.

### Phase 2: Fairness and Bias Mitigation
* **Farah - Audit (Pre-Mitigation):** Using `Fairlearn` or `AIF360` to measure model disparities (e.g., *Disparate Impact*) regarding sensitive attributes (Sex/Age) and plotting the initial bias metrics.
* **Olivia - Mitigation (Post-Mitigation):** Implementing the *Reweighting* technique (adjusting weights for training instances), retraining the PyTorch model, and validating that the bias has been successfully reduced.

### Phase 3: Robustness (Adversarial Machine Learning)
* **Farah - The Attack (FGSM):** Implementing the *Fast Gradient Sign Method* using the `ART` (Adversarial Robustness Toolbox) library to generate adversarial examples and demonstrating the drop in model accuracy.
* **Olivia - The Defense (Adversarial Training):** Integrating Farah's adversarial examples into the PyTorch training loop to train a new, robust model capable of resisting the FGSM attack.

### Phase 4: Data Privacy
* **Olivia - The Attack (MIA):** Implementing a *Membership Inference Attack* via `ART` to guess which data points belong to the training set, measuring the attack's success rate (data leakage).
* **Farah - The Defense (DP-SGD):** Utilizing the `Opacus` library in PyTorch to apply *Differential Privacy* (gradient clipping and noise addition) during training, effectively reducing the MIA success rate to ~50% (random guessing).

### Phase 5: Synthesis and Reporting
* **Together:** Analyzing the "Trade-offs" (Accuracy vs. Fairness, Accuracy vs. Robustness, Accuracy vs. Privacy) and building the final comparative table.
* **Writing (Farah):** Technical justifications for data preparation, bias auditing, the FGSM attack, and the DP-SGD defense.
* **Writing (Olivia):** Technical justifications for the PyTorch baseline model, bias mitigation, Adversarial Training defense, and the MIA attack.

## 🛠️ Setup and Execution
*(To be completed as development progresses)*
```bash
# Clone the repository
git clone [https://github.com/your-username/trustworthy-ai-project.git](https://github.com/your-username/trustworthy-ai-project.git)

# Install the required dependencies
pip install -r requirements.txt

# Run the main pipeline
python src/main.py