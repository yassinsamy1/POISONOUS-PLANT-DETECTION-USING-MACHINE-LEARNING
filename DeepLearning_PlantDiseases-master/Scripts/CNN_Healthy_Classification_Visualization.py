# Visualize the CNN process (without graphviz) using matplotlib

import matplotlib.pyplot as plt

def visualize_cnn_process_diagram():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Box positions and labels (organized for better alignment)
    boxes = [
        (0.08, 0.7, "Step 1:\nInput Leaf Image"),
        (0.22, 0.7, "Step 2:\nPreprocessing\n(Resize, Normalize)"),
        (0.36, 0.7, "Step 3:\nConvolutional Filters\n(Feature Extraction)"),
        (0.50, 0.7, "Step 4:\nFlatten & Dense Layers"),
        (0.64, 0.7, "Step 5:\nTrained CNN Model"),
        (0.78, 0.7, "Step 6:\nModel Outputs Probabilities"),
        (0.78, 0.45, "Step 7:\nDecision:\nHealthy or Not Healthy?"),
        (0.64, 0.2, "Class: Healthy"),
        (0.92, 0.2, "Class: Not Healthy"),
    ]

    # Draw boxes
    for x, y, text in boxes:
        if "Decision" in text:
            fc = "lightyellow"
        elif "Healthy" in text and "Not" not in text:
            fc = "lightgreen"
        elif "Not Healthy" in text:
            fc = "lightcoral"
        else:
            fc = "white"
        ax.text(x, y, text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec="black"))

    # Remove arrows, just show the steps in order
    for i in range(len(boxes)-1):
        x1, y1, _ = boxes[i]
        x2, y2, _ = boxes[i+1]
        # Only draw arrows for the last two transitions (step 7 to healthy/not healthy)
        if i == 6:
            # Arrow from step 7 (decision) to healthy
            ax.annotate("", xy=(0.64, 0.25), xytext=(0.78, 0.45), arrowprops=dict(arrowstyle="->", color='black', lw=2))
            ax.text(0.66, 0.33, "Yes", ha='center', va='center', fontsize=11, color='green')
            # Arrow from step 7 (decision) to not healthy
            ax.annotate("", xy=(0.92, 0.25), xytext=(0.78, 0.45), arrowprops=dict(arrowstyle="->", color='black', lw=2))
            ax.text(0.90, 0.33, "No", ha='center', va='center', fontsize=11, color='red')
        else:
            # Draw step arrows for all other transitions
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y+0.07, f"→", ha='center', va='center', fontsize=18, color='gray')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_cnn_process_diagram()
