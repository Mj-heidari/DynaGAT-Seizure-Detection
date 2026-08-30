
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_pr(results, out):
    # Replace with stored prediction arrays after training
    x=np.linspace(0,1,100)
    plt.figure(figsize=(5,5))
    plt.plot(x, x)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - DynaGAT-Onset")
    plt.savefig(Path(out)/"Figure2_ROC.pdf",bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5,5))
    plt.plot(x, 1-np.sqrt(x))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - DynaGAT-Onset")
    plt.savefig(Path(out)/"Figure2_PR.pdf",bbox_inches="tight")
    plt.close()
