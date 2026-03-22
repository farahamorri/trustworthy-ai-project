import torch
import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate
import matplotlib.pyplot as plt


def evaluate_fairness(model, X_test, y_test):
    """Audits the model for Demographic Parity between Men and Women."""
    print("\n⚖️ Starting Fairness Audit")
    
    # 1. Generate model predictions
    model.eval()
    X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        # Flatten to get a 1D array of 0s and 1s
        predictions = torch.sigmoid(outputs).round().numpy().flatten()
        
    y_true = y_test.values

    # 2. Reconstruct the sensitive attribute (Sex)
    # Because of StandardScaler: Sex = 1 (Men) is < 0, Sex = 2 (Women) is > 0
    sensitive_feature = (X_test['SEX'] > 0).astype(int) 
    # Mapping: 0 = Men, 1 = Women
    
    # 3. Calculate metrics with Fairlearn
    # "selection_rate" measures the % of times the model predicts "1" (Default risk)
    mf = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=predictions,
        sensitive_features=sensitive_feature
    )
    
    rate_men = mf.by_group[0] * 100
    rate_women = mf.by_group[1] * 100
    
    print(f"Prediction rate of 'Default' for Men : {rate_men:.2f}%")
    print(f"Prediction rate of 'Default' for Women : {rate_women:.2f}%")
    
    # Demographic Parity Difference (Absolute gap between the two groups)
    dpd = demographic_parity_difference(y_true, predictions, sensitive_features=sensitive_feature)
    print(f"🛑 Demographic Parity Difference: {dpd * 100:.2f}%")
    
    if dpd > 0.05:
        print("⚠️ ALERT: The model is biased. The gap between genders exceeds 5%.")
    else:
        print("✅ The model is considered fair (gap < 5%).")
        
    return mf


def get_sample_weights(X_train, y_train):
    """Calculates instance weights using the Kamiran & Calders mitigation technique."""
    print("\n⚖️ Calculating mitigation weights (Kamiran & Calders)")
    
    sensitive_attr = (X_train['SEX'] > 0).astype(int).values
    y = y_train.values
    N = len(y)
    
    weights = np.zeros(N)
    
    for a in [0, 1]:
        for target in [0, 1]:
            # Find the masks
            mask_a = (sensitive_attr == a)
            mask_y = (y == target)
            mask_ay = mask_a & mask_y
            
            # Count occurrences
            count_a = mask_a.sum()
            count_y = mask_y.sum()
            count_ay = mask_ay.sum()
            
            # Apply the exact mathematical formula
            if count_ay > 0:
                weights[mask_ay] = (count_a * count_y) / (N * count_ay)
                
    # Return as a PyTorch tensor with shape [N, 1]
    return torch.tensor(weights, dtype=torch.float32).unsqueeze(1)



def plot_fairness_comparison(mf_before, mf_after):
    """
    Plots a side-by-side bar chart comparing the selection rates 
    (predictions of Default) for Men and Women, before and after mitigation.
    """
    print("\n📊 Generating Fairness Comparison Chart...")
    
    # 1. Extract data from the MetricFrames
    # Group 0 = Men, Group 1 = Women
    rates_before = [mf_before.by_group[0] * 100, mf_before.by_group[1] * 100]
    rates_after = [mf_after.by_group[0] * 100, mf_after.by_group[1] * 100]
    
    # Calculate Demographic Parity Difference (Gap) for text display
    gap_before = abs(rates_before[0] - rates_before[1])
    gap_after = abs(rates_after[0] - rates_after[1])

    # 2. Set up the plot
    labels = ['Men (Group 0)', 'Women (Group 1)']
    x = np.arange(len(labels))  # Label locations
    width = 0.35  # Bar width
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the bars
    rects1 = ax.bar(x - width/2, rates_before, width, label='Standard Model (Biased)', color='salmon')
    rects2 = ax.bar(x + width/2, rates_after, width, label='Fair Model (Mitigated)', color='skyblue')

    # 3. Add text, labels, and titles
    ax.set_ylabel('% Predicted as Default (Selection Rate)')
    ax.set_title('Impact of Fairness Mitigation (Kamiran & Calders)\nPrediction Rates by Gender', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # 4. Add data labels on top of the bars
    def autolabel(rects):
        """Attach a text label above each bar, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # 5. Add a text box to summarize the gap reduction
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    summary_text = (f"Gap Before: {gap_before:.1f}%\n"
                    f"Gap After: {gap_after:.1f}%")
    ax.text(0.5, -0.15, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='center', bbox=props)

    plt.tight_layout()
    plt.show()