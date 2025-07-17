import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrices(y_trues, y_preds, model_names, labels=None, figsize=(5, 5)):
    """
    Plots confusion matrices for multiple models side by side.
    y_trues: list of true label arrays
    y_preds: list of predicted label arrays
    model_names: list of model names (strings)
    labels: list of class labels (optional)
    """
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0]*n, figsize[1]))
    if n == 1:
        axes = [axes]
    for i in range(n):
        cm = confusion_matrix(y_trues[i], y_preds[i], labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(model_names[i])
    plt.tight_layout()
    plt.show()

def plot_model_metrics(metrics_dict):
    """
    Plots a bar chart comparing model metrics.
    metrics_dict: dict of {model_name: {'accuracy': val, 'precision': val, ...}}
    """
    import pandas as pd
    df = pd.DataFrame(metrics_dict).T
    # Greatly increase figure size for both axes
    fig, ax = plt.subplots(figsize=(10 * len(df.columns), 20))  # Expand x and y axes
    df.T.plot(kind='bar', ax=ax)
    plt.ylabel('Score', fontsize=24)
    plt.xlabel('Metric', fontsize=24)
    plt.title('Model Metrics Comparison', fontsize=28)
    ax.set_ylim(bottom=0)  # y-axis starts at 0, upper limit auto
    plt.legend(title='Model', loc='lower right', fontsize=18, title_fontsize=20)
    ax.set_xticklabels([label for label in df.columns], rotation=90, ha='center', fontsize=20)  # Larger font
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout(pad=5.0)
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    import sys

    # Read stats.csv
    stats_path = "d:\\Final_Year_Project\\DeepLearning_PlantDiseases-master\\DeepLearning_PlantDiseases-master\\Scripts\\stats.csv"
    df = pd.read_csv(stats_path)

    # Check for required columns
    required_columns = {'accuracy', 'precision', 'recall'}
    missing = required_columns - set(df.columns)
    if missing:
        print(f"Error: The following required columns are missing in stats.csv: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Prepare data for confusion matrices
    y_trues = []
    y_preds = []
    model_names = []
    metrics = {}
    labels_set = set()

    # Check if 'model', 'y_true', 'y_pred', 'f1' columns exist before using them
    has_model = 'model' in df.columns
    has_y_true = 'y_true' in df.columns
    has_y_pred = 'y_pred' in df.columns
    has_f1 = 'f1' in df.columns

    for idx, row in df.iterrows():
        # Use index as model name if 'model' column is missing
        model = row['model'] if has_model else f"Model_{idx+1}"
        # Only process confusion matrix if both y_true and y_pred exist
        if has_y_true and has_y_pred:
            y_true = [int(x) for x in str(row['y_true']).split(',')]
            y_pred = [int(x) for x in str(row['y_pred']).split(',')]
            y_trues.append(y_true)
            y_preds.append(y_pred)
            model_names.append(model)
            labels_set.update(y_true)
            labels_set.update(y_pred)
        # Always collect metrics
        metrics[model] = {
            'accuracy': float(row['accuracy']),
            'precision': float(row['precision']),
            'recall': float(row['recall']),
        }
        if has_f1:
            metrics[model]['f1'] = float(row['f1'])

    # Only plot confusion matrices if data is available
    if y_trues and y_preds and model_names:
        labels = sorted(labels_set)
        plot_confusion_matrices(y_trues, y_preds, model_names, labels=labels)
    else:
        print("Skipping confusion matrix plot: 'y_true' or 'y_pred' columns not found in stats.csv.")

    plot_model_metrics(metrics)
