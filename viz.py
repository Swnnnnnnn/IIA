import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("./storage/model.pkl", "rb") as f:
    result = pickle.load(f)

print("Mean absolute correlation:", result['meanabscorr_tr'])
print("Correlation matrix:")
print(np.round(result['corrmat_tr'], 2))

corr = result["corrmat_tr"]

# Plot
plt.figure(figsize=(6, 5))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xlabel("Estimated innovations")
plt.ylabel("True innovations")
plt.title("Correlation between true and estimated innovations")

plt.tight_layout()
plt.savefig("corr_matrix.pdf")
plt.savefig("corr_matrix.png", dpi=300)
plt.show()