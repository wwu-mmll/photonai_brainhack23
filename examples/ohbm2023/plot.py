import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

root_folder = Path('./tmp')

plt.figure()
classification_data = pd.read_csv(root_folder.joinpath("classification.csv"))
sns.barplot(data=classification_data, x="analysis", y="f1_score_mean", hue="config_selector_name")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.show()


regression_data = pd.read_csv(root_folder.joinpath("regression.csv"))
for a in regression_data["analysis"].unique():
    plt.figure()
    plot_df = regression_data[regression_data["analysis"] == a]
    x = [a] * plot_df.shape[0]
    sns.barplot(data=plot_df, x=x, y="mean_squared_error_mean", hue="config_selector_name")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()

