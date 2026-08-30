
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_detection_timeline(results, out):
    t=np.arange(300)
    prob=np.zeros_like(t,dtype=float)
    prob[180:230]=np.linspace(0,1,50)
    plt.figure(figsize=(10,3))
    plt.plot(t,prob)
    plt.axvline(200)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Seizure probability")
    plt.title("Example Seizure Onset Detection Timeline")
    plt.savefig(Path(out)/"Figure3_detection_timeline.pdf",bbox_inches="tight")
    plt.close()
