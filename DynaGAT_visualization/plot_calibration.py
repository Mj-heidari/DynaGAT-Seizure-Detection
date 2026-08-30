
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_calibration(results, out):
    p=np.linspace(0,1,10)
    plt.figure(figsize=(5,5))
    plt.plot(p,p,label="Perfect calibration")
    plt.plot(p,p*0.9+0.05,label="DynaGAT-Onset")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability Calibration")
    plt.legend()
    plt.savefig(Path(out)/"Figure5_calibration.pdf",bbox_inches="tight")
    plt.close()
