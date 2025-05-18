import numpy as np
import matplotlib.pyplot as plt

train_images = np.load('quickdraw_subset_np/train_images.npy')
train_labels = np.load('quickdraw_subset_np/train_labels.npy')
test_images = np.load('quickdraw_subset_np/test_images.npy')
test_labels = np.load('quickdraw_subset_np/test_labels.npy')

train_images = train_images[(train_labels==0) | (train_labels==2)]
train_labels = train_labels[(train_labels==0) | (train_labels==2)]
test_images = test_images[(test_labels==0) | (test_labels==2)]
test_labels = test_labels[(test_labels==0) | (test_labels==2)]

train_labels[train_labels == 0] = -1
train_labels[train_labels == 2] = 1
test_labels[test_labels == 0] = -1
test_labels[test_labels == 2] = 1
train_flat = train_images.reshape(train_images.shape[0], -1)
test_flat = test_images.reshape(test_images.shape[0], -1)
train_labels = train_labels.reshape(train_labels.shape[0], 1)
test_labels = test_labels.reshape(test_labels.shape[0], 1)
# PCA
#https://medium.com/technological-singularity/build-a-principal-component-analysis-pca-algorithm-from-scratch-7515595bf08b


def PCA_from_Scratch(X, n_components):
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-4)
    cov_mat = np.cov(X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)  # Use eigh for symmetric matrices (covariance is symmetric)
    sort_indices = np.argsort(eigen_values)[::-1]
    principal_components = eigen_vectors[:, sort_indices[:n_components]]

    return principal_components


def transform(X, principal_components):
    X = X.copy()
    return X.dot(principal_components)



# LDA 
#https://www.kaggle.com/code/egazakharenko/linear-discriminant-analysis-lda-from-scratch
class LDA():
  def __init__(self, n_components=None):
     self.n_components = n_components
  def fit(self,X,y):
     self.X = X
     self.y = y
     samples = X.shape[0]
     features= X.shape[1]
     classes, cls_counts = np.unique(y,return_counts=True)
     priors = cls_counts/samples
     X_mean = np.array([X[y==cls].mean(axis=0) for cls in classes])
     betweenCLSdeviation = X_mean - X.mean(axis=0)
     withinCLSdeviation = X - X_mean[y]

     Sb = priors* betweenCLSdeviation.T @ betweenCLSdeviation
     Sw = withinCLSdeviation.T @ withinCLSdeviation / samples
     Sw_inv = np.linalg.pinv(Sw)
     eigvals, eigvecs = np.linalg.eig(Sw_inv @ Sb)
     self.dvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
     self.weights = X_mean @ self.dvecs @ self.dvecs.T
     self.bias = np.log(priors) - 0.5 * np.diag(X_mean @ self.weights.T)
     if self.n_components is None:
        self.n_components = min(classes.size - 1, features)
  def transform(self, X):
    return X @ self.dvecs[:, : self.n_components]

  def predict(self, X_test):
    scores = X_test @ self.weights.T + self.bias
    return np.argmax(scores, axis=1)
  
def score(y_pred, y_true):
        accuracy = np.mean(y_pred == y_true)
        return accuracy

def macroF1(y_pred, y_true):
    labels = np.unique(y_true)
    f1_scores = []

    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    return np.mean(f1_scores)
train_normal = train_flat/255
test_normal = test_flat/255
pcs = PCA_from_Scratch(train_normal, 128)
train_proj = transform(train_normal, pcs)
test_proj = transform(test_normal, pcs)
train_binary = (train_normal > 0.3).astype(int)
test_binary = (test_normal > 0.3).astype(int)
train_labels = np.expand_dims(train_labels,axis=1)




kernels = ['linear','poly','rbf']
Cs = [0.1,0.125, 0.15,0.175, 0.2]
feature_sets = {
    'normal': (train_normal, test_normal),
    'binary': (train_binary, test_binary)
}
test_labels = test_labels.flatten()
PCAlist = [32, 64, 128, 256]
for dim in PCAlist:
    print("PCA: ", dim)
    pcs = PCA_from_Scratch(train_normal, dim)
    train_proj = transform(train_normal, pcs)
    test_proj = transform(test_normal, pcs)
    feature_sets[f'pca{dim}'] = (train_proj, test_proj)
param_grid = {'C': Cs}
results = {}
def crossValidation(X, y, k=5):
    from collections import defaultdict

    y = y.flatten()
    label_indices = defaultdict(list)
    for idx, label in enumerate(y):
        label_indices[label].append(idx)

    folds = [[] for _ in range(k)]
    for indices in label_indices.values():
        np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    trainsets = []
    for i in range(k):
        idxs = np.array(folds[i])
        trainsets.append((X[idxs], y[idxs]))

    return trainsets
import time
from sklearn.svm import SVC
for name, (X_tr, X_te) in feature_sets.items():
    print(f"\n=== Feature set: {name} ===")
    # build your 5 folds once
    folds = crossValidation(X_tr, train_labels)

    for kernel in kernels:
        print(f"\n-- Kernel: {kernel} --")
        for C in Cs:
            fold_scores = []
            start = time.time()

            for fold_idx in range(5):
                val_X, val_y = folds[fold_idx]
                val_y = val_y.flatten()

                # train on the other 4 folds
                train_X = np.vstack([folds[j][0] for j in range(5) if j != fold_idx])
                train_y = np.hstack([folds[j][1] for j in range(5) if j != fold_idx]).flatten()

                # choose the right SVC
                if kernel == 'linear':
                    clf = SVC(kernel='linear', C=C)
                elif kernel == 'rbf':
                    clf = SVC(kernel='rbf',    C=C, gamma='scale')
                else:  # 'poly'
                    clf = SVC(kernel='poly',   C=C,
                              degree=3, coef0=1, gamma='scale')

                clf.fit(train_X, train_y)
                preds = clf.predict(val_X)
                acc = score(val_y, preds)
                print(f"{kernel} | C={C:.3f} | fold={fold_idx+1} | Acc={acc:.4f}")
                fold_scores.append(acc)

            cv_time = time.time() - start
            avg_acc = np.mean(fold_scores)
            print(f"-> {kernel} | C={C:.3f} | AvgCV_Acc={avg_acc:.4f} | Time={cv_time:.1f}s")

            # store for later
            results[(name, kernel, C)] = {
                'fold_accs':  fold_scores,
                'avg_cv_acc': avg_acc,
                'cv_time_s':  cv_time
            }
pcs = PCA_from_Scratch(train_normal, dim)
train_proj = transform(train_normal, pcs)
test_proj = transform(test_normal, pcs)
clf = SVC(kernel='poly',   C=C,
                              degree=3, coef0=1, gamma='scale')

clf.fit(train_proj, train_labels.flatten())
sv_indices = clf.support_
dists    = clf.decision_function(train_proj)
abs_dists = np.abs(dists)
far_all  = np.argsort(abs_dists)[-len(sv_indices):]
dists = clf.decision_function(train_proj)          
abs_dists = np.abs(dists)

sv_idx  = sv_indices[:3]
far_idx = far_all[:3]

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for ax, idx in zip(axes[0], sv_idx):
    ax.imshow(train_images[idx], cmap='gray')
    ax.axis('off')
    ax.set_title(f"SV idx={idx}")
for ax, idx in zip(axes[1], far_idx):
    ax.imshow(train_images[idx], cmap='gray')
    ax.axis('off')
    ax.set_title(f"Far idx={idx}")

plt.tight_layout()
plt.show()