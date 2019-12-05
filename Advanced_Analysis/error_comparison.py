import matplotlib.pyplot as plt
import pandas as pd

dpi = 300

percent_residual_kNN_L1 = pd.read_csv("kNN_L1_percent_residual.csv")
percent_residual_RF = pd.read_csv("RF_percent_residual.csv")
percent_residual_FFNN = pd.read_csv("FFNN_percent_residual.csv")

residuals = pd.DataFrame()
residuals["kNN"] = percent_residual_kNN_L1
residuals["RF"] = percent_residual_RF
residuals["FFNN"] = percent_residual_FFNN

plt.figure()
plt.scatter(residuals["RF"], residuals["kNN"], s=4, alpha=.5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("RF Absolute Percent Error")
plt.ylabel("kNN with L1 Absolute Percent Error")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_xticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_xticks()])
plt.gca().set_yticklabels(['{:.0f}%'.format(y * 100) for y in plt.gca().get_yticks()])
plt.title("kNN with L1 vs RF Absolute Percent Error (Capped at 100%)")
plt.tight_layout()
plt.savefig(f"figures/kNN_with_L1_vs_RF_Absolute_Percent_Error.png", dpi=dpi)


plt.figure()
plt.scatter(residuals["RF"], residuals["FFNN"], s=4, alpha=.5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("RF Absolute Percent Error")
plt.ylabel("FFNN Absolute Percent Error")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_xticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_xticks()])
plt.gca().set_yticklabels(['{:.0f}%'.format(y * 100) for y in plt.gca().get_yticks()])
plt.title("FFNN vs RF Absolute Percent Error (Capped at 100%)")
plt.tight_layout()
plt.savefig(f"figures/FFNN_vs_RF_Absolute_Percent_Error.png", dpi=dpi)


plt.figure()
plt.scatter(residuals["FFNN"], residuals["kNN"], s=4, alpha=.5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FFNN Absolute Percent Error")
plt.ylabel("kNN with L1 Absolute Percent Error")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_xticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_xticks()])
plt.gca().set_yticklabels(['{:.0f}%'.format(y * 100) for y in plt.gca().get_yticks()])
plt.title("kNN with L1 vs FFNN Absolute Percent Error (Capped at 100%)")
plt.tight_layout()
plt.savefig(f"figures/kNN_with_L1_vs_FFNN_Absolute_Percent_Error.png", dpi=dpi)
