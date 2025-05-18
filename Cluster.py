import numpy as np

train_images = np.load('quickdraw_subset_np/train_images.npy')
train_labels = np.load('quickdraw_subset_np/train_labels.npy')
test_images = np.load('quickdraw_subset_np/test_images.npy')
test_labels = np.load('quickdraw_subset_np/test_labels.npy')


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

pcs = PCA_from_Scratch(train_flat, 128)
train_proj = transform(train_flat, pcs)
test_proj = transform(test_flat, pcs)
train_normal = train_flat/255
test_normal = test_flat/255


class KMeansClustering:
    def __init__(self, k, max_iter=1000,distance_metric='euclidean'):
        self.k = k
        self.max_iter = max_iter
        self.distance_metric = distance_metric
    
    def euclidean_distance(self, a, b):
        return np.linalg.norm(a - b, axis=0)
    
    def _manhattan(self, p1, p2):
        return np.sum(np.abs(p1 - p2))

    def cosine_distance(self, a, b):  
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 1.0  # maximal distance if one vector is zero
        return 1 - np.dot(a, b) / denom
    
    def _distance(self, a, b):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(a, b)
        elif self.distance_metric == 'manhattan':
            return self._manhattan(a, b)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(a, b)
        
    def predict(self, X):
        distances = np.array([[self._distance(x, c) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.centroids = X[np.random.choice(self.n_samples, self.k, replace=False)]
        self.labels = np.zeros(self.n_samples)
        
        for _ in range(self.max_iter):
            distances = np.array([[self._distance(x, c) for c in self.centroids] for x in X])
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            if np.all(new_centroids == self.centroids):
                break
            
            self.centroids = new_centroids
        
        return self
    

print("Starting...")
km = KMeansClustering(k=5, max_iter=1000, distance_metric='euclidean')
km.fit(train_normal)
train_clusters = km.labels   

mapping = {}
for c in range(km.k):
    mask = (train_clusters == c)
    if not np.any(mask):
        mapping[c] = -1
    else:
        lbls, counts = np.unique(train_labels[mask], return_counts=True)
        mapping[c] = lbls[np.argmax(counts)]

test_clusters = km.predict(test_normal)
y_pred = np.array([ mapping[c] for c in test_clusters ])

print("KMeans Accuracy:     ", score(y_pred, test_labels))
print("KMeans Macro F1:     ", macroF1(y_pred, test_labels))

print(y_pred)