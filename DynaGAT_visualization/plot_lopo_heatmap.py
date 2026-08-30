
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_lopo_heatmap(csv_file, out):
    df = pd.read_csv(csv_file)
    numeric = df.select_dtypes("number")
    plt.figure(figsize=(10,6))
    plt.imshow(numeric.T, aspect="auto")
    plt.yticks(range(len(numeric.columns)), numeric.columns)
    plt.xticks(range(len(df)), df.iloc[:,0], rotation=45)
    plt.colorbar(label="value")
    plt.title("LOPO Patient-Level Performance")
    plt.tight_layout()
    plt.savefig(Path(out)/"Figure1_LOPO_heatmap.pdf", bbox_inches="tight")
    plt.close()
