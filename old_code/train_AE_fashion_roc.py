import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

cwd = os.getcwd()

betas_vae = 0.0128
betas_rae = 0.38909
#betas = np.array([0, 0.01])

frac_anom = np.array([0.1])
# frac_anom = np.array([0.05])

FPRs = dict()
TPRs = dict()
AUC = dict()

lgd = {
    0: 'RVAE-10%',
    1: 'RRAE-10%',
   
}
colors = {0: 'r', 1: 'b'}
lsty = {0: '--', 1: '-'}
c = 0
beta_str_vae = str(betas_vae)
beta_str_rae = str(betas_rae)
for frac in frac_anom:
    filename = cwd + '/results/fashion_mnist_' + beta_str_vae + '_' + str(
            frac) + '.npz'
    filename1 = cwd + '/results/fashion_rae_mnist_' + beta_str_rae + '_' + str(
            frac) + '.npz'

    s = np.load(filename)

    y = s['recon']
    x = s['data']
    L = s['anom_lab']

    y = y.reshape(y.shape[0], -1)
    x = x.reshape(x.shape[0], -1)
    L = L.reshape(L.shape[0], -1)

    mse = np.linalg.norm(x - y, 2, 1, True)
    fpr, tpr, _ = roc_curve(L, mse)
    auc = roc_auc_score(L,mse)

    FPRs[c] = fpr
    TPRs[c] = tpr
    AUC[c] = auc

    c = c + 1


    s = np.load(filename1)

    y = s['recon']
    x = s['data']
    L = s['anom_lab']

    y = y.reshape(y.shape[0], -1)
    x = x.reshape(x.shape[0], -1)
    L = L.reshape(L.shape[0], -1)

    mse = np.linalg.norm(x - y, 2, 1, True)
    fpr, tpr, _ = roc_curve(L, mse)
    auc = roc_auc_score(L,mse)

    FPRs[c] = fpr
    TPRs[c] = tpr
    AUC[c] = auc
    c=c+1
        # print(c)

lw = 2
fig = plt.figure(figsize=(8, 6), dpi=300)

for c in np.arange(0, len(FPRs)):
    fpr = FPRs[c]
    tpr = TPRs[c]

    plt.plot(fpr, tpr, lsty[c], color=colors[c], lw=lw, label=lgd[c])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    # print(c)

#plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.5, 1.05])
plt.legend()
plt.show()

fig.savefig('ROCn_fashion_mnist_bernoulli_v2.png')

print(AUC)