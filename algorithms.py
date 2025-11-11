import numpy as np
import pandas as pd
from scipy import spatial

# ----------------- Reading in the data -----------------
def read_data(trainfile ='MNIST_train.csv', validationfile ='MNIST_validation.csv'):
    
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'label'

    Xtrain = dftrain[featurecols]
    ytrain = dftrain[targetcol]
    
    Xval = dfval[featurecols]
    yval = dfval[targetcol]

    return (Xtrain, ytrain, Xval, yval)

# ----------------- Constructing the PCA Model -----------------
class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        # Initialize all these as None
        self.mean = None
        self.std = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):

        # Fit PCA on the dataset X.
        # Convert to array
        X = np.array(X, dtype=float)
        
        # Center the data
        # compute mean 
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=1)
        self.std[self.std == 0] = 1.0

        # center around mean
        X_centered = (X - self.mean) / self.std

        # Covariance matrix
        # compute variance - each feature in columns - rowvar is False
        # rowvar False means features are columns, True means features are rows
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        # compute the eigen values and vectors of covariance martix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by descending eigenvalues
        # sort in descending order of eigen values
        sorted_idx = np.argsort(eigenvalues)[::-1]

        # take top n components
        self.explained_variance = eigenvalues[sorted_idx][:self.n_components]

        # get the components 
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

    def predict(self, X):

        # Project the data X onto the principal components.
        if self.mean is None or self.components is None:
            # eigen values or vectors do not exist if model is not fitted yet
            raise ValueError("The PCA model has not been fitted yet.")

        # center the data around zero
        X_centered = (X - self.mean) / self.std
        
        # return the dot product with eigen vectors, to retrieve components
        return np.dot(X_centered, self.components)

    def reconstruct(self, X):
        # Reconstruct the original data from the reduced representation.
        # predict projections
        Z = self.predict(X)  # Projected data
        
        # reconstruct X based on projects onto components, and add mean value 
        return np.dot(Z, self.components.T) * self.std + self.mean
    
# ----------------- Constructing the Softmax Classifier -----------------
def softmax(z):
    # z.shape = (n_samples, n_classes)
    # Subtract max for numerical stability to prevent overflow
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def to_one_hot(y, n_classes):
    m = y.shape[0]
    one_hot_y = np.zeros((m, n_classes))
    one_hot_y[np.arange(m), y] = 1
    return one_hot_y

def compute_categorical_cross_entropy(Y_true_one_hot, Y_pred_proba):
    epsilon = 1e-15
    Y_pred_proba = np.clip(Y_pred_proba, epsilon, 1 - epsilon)
    # Loss for each sample
    loss_per_sample = -np.sum(Y_true_one_hot * np.log(Y_pred_proba), axis=1)
    # Mean loss over the batch
    return np.mean(loss_per_sample)

def predict(x, theta):
    x = np.array(x)
    m = len(x)
    # Add bias term
    x_b = np.c_[np.ones((m, 1)), x]
    # Compute logits
    logits = np.dot(x_b, theta)
    # Compute probabilities
    probabilities = softmax(logits)
    # Get class prediction
    ypred = np.argmax(probabilities, axis=1)
    return (probabilities, ypred)

def Softmax_Regression(x, y, n_classes = 10, mini_batch_size = 100, learning_rate = 0.01, n_epochs=1000):
    def add_bias(x):
        x = np.array(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        m = x.shape[0]
        x_b = np.c_[np.ones((m, 1)), x] 
        return x_b, m, x.shape[1]

    x_b, m, n = add_bias(x)
    y_one_hot = to_one_hot(y, n_classes) # (m, n_classes)
    # Initialize theta as (n+1, n_classes)
    np.random.seed(42) # for reproducibility
    theta = np.random.randn(n + 1, n_classes) * 0.01 # Initialising theta to small values
    logloss = []
    n_batches = m // mini_batch_size

    for _ in range(n_epochs):
        indices = np.random.permutation(m)
        x_shuffled = x_b[indices]
        y_shuffled = y_one_hot[indices]
        for i in range(n_batches):
            start_idx = i * mini_batch_size
            end_idx = start_idx + mini_batch_size
            x_batch = x_shuffled[start_idx : end_idx]
            y_batch = y_shuffled[start_idx : end_idx]

            logits = np.dot(x_batch, theta)
            predictions = softmax(logits)

            errors = (y_batch - predictions)
            gradients = (-1.0 / mini_batch_size) * np.dot(x_batch.T, errors)
            theta -= learning_rate * gradients
        
        full_logits = np.dot(x_b, theta)
        full_predictions = softmax(full_logits)
        loss = compute_categorical_cross_entropy(y_one_hot, full_predictions)
        logloss.append(loss)       
    return (theta, logloss)

# Performing bootstrap aggregation for the Softmax Classifier
class BaggingSoftmaxClassifier:
    def __init__(self, n_estimators,  **base_model_params): # **base_model_params collects all other parameters (model parameters) passed
        self.n_estimators = n_estimators # Number of bootstraps
        self.base_model_params = base_model_params
        self.models = [] # To store trained theta for each model
        self.n_classes = None
        self.PCA_model = None # Because using all 784 features will take very long

    def fit(self, X, y):
        self.PCA_model = PCAModel(n_components= 100)
        self.PCA_model.fit(X)
        X = self.PCA_model.predict(X) # PCA on training dataset
        
        m_samples, _ = X.shape
        self.n_classes = len(np.unique(y))
        self.models = []
        print(f"  Beggining bagging for the Softmax Classifier with: {self.base_model_params}")

        for i in range(self.n_estimators):
            print(f"    Training model {i+1}/{self.n_estimators}...")
            indices = np.random.choice(m_samples, size = m_samples, replace = True) # sampling with replacement to form our bootstraps
            X_boot = X[indices]
            y_boot = y[indices]
            theta, _ = Softmax_Regression(X_boot, y_boot, n_classes=self.n_classes, **self.base_model_params)
            self.models.append(theta)

    def predict_proba(self, X):
        if not self.models:
            raise ValueError("Model has not been fit yet.")
        
        X = self.PCA_model.predict(X) # PCA on validation dataset
        
        all_probas = []
        for theta in self.models:
            probas, _ = predict(X, theta)
            all_probas.append(probas)

        stacked_probas = np.stack(all_probas) # This will be an array of dimensions (n_estimators, len(theta))
        avg_probas = np.mean(stacked_probas, axis=0) # Averaging probabilites across bootstraps
        return avg_probas

    def predict(self, X):
        avg_probas = self.predict_proba(X)
        return np.argmax(avg_probas, axis=1) # Returns class that has the maximum average probability
    
# ----------------- Constructing the XGBoost Classifier -----------------
class Node:
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class XGBTree:
    def __init__(self, max_depth = 3, lam = 1.0, gamma = 0.0, min_child_weight = 1):
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.root = None

    def fit(self, X, g, h):
        self.root = self._build_tree(X, g, h, depth=0)
        return self

    def _build_tree(self, X, g, h, depth):
        G, H = np.sum(g), np.sum(h)
        if depth >= self.max_depth or len(X) < 2:
            return Node(value=-G / (H + self.lam))

        _, n_features = X.shape
        best_gain, best_feat, best_thresh = -np.inf, None, None

        for j in range(n_features):
            order = np.argsort(X[:, j])
            Xj, gj, hj = X[order, j], g[order], h[order]

            G_cumsum = np.cumsum(gj)
            H_cumsum = np.cumsum(hj)
            G_total, H_total = G_cumsum[-1], H_cumsum[-1]

            unique_mask = np.diff(Xj) > 1e-12
            if not np.any(unique_mask):
                continue

            G_L = G_cumsum[:-1][unique_mask]
            H_L = H_cumsum[:-1][unique_mask]
            G_R = G_total - G_L
            H_R = H_total - H_L

            valid = (H_L >= self.min_child_weight) & (H_R >= self.min_child_weight)
            if not np.any(valid):
                continue

            gain = 0.5 * ((G_L[valid] ** 2 / (H_L[valid] + self.lam)) + (G_R[valid] ** 2 / (H_R[valid] + self.lam))
             - (G_total ** 2 / (H_total + self.lam))) - self.gamma

            if gain.size > 0:
                idx = np.argmax(gain)
                if gain[idx] > best_gain:
                    best_gain = gain[idx]
                    best_feat = j
                    best_thresh = (Xj[:-1][unique_mask][valid])[idx]

        if best_feat is None or best_gain <= 1e-6:
            return Node(value=-G / (H + self.lam))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left_node = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)

        return Node(feature_index=best_feat, threshold=best_thresh, left=left_node, right=right_node)

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
    
class XGBoostMultiClassifier:
    def __init__(self, n_classes = 10, n_estimators = 100, learning_rate = 0.1, max_depth = 3, lam = 1.0, gamma = 0.1, colsample_bytree = 1.0, subsample = 1.0):
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.trees = []

    def _softmax(self, logits):
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        n_samples, n_features = X.shape
        K = self.n_classes
        y_pred = np.zeros((n_samples, K))

        print(f"  Training the XGBoost Classifier with n_estimators: {self.n_estimators}, learning_rate: {self.learning_rate}, max_depth: {self.max_depth}, colsample_bytree: {self.colsample_bytree}, subsample: {self.subsample}")
        for i in range(self.n_estimators):
            print(f"    Building tree {i + 1}/{self.n_estimators}...")
            # Row sampling
            if self.subsample < 1.0:
                idx = np.random.choice(n_samples, size=int(n_samples * self.subsample), replace=False)
                X_round, _ = X[idx], y[idx]
            else:
                X_round, _ = X, y

            # Computing gradients and Hessians
            p = self._softmax(y_pred)
            g = np.zeros_like(p)
            h = np.zeros_like(p)
            for k in range(K):
                g[:, k] = p[:, k] - (y == k)
                h[:, k] = p[:, k] * (1 - p[:, k])

            trees_this_round = []
            for k in range(K):
                # Feature sampling
                if self.colsample_bytree < 1.0:
                    feat_mask = np.random.choice(n_features, size=int(n_features * self.colsample_bytree), replace=False)
                    X_sub = X_round[:, feat_mask]
                else:
                    feat_mask = np.arange(n_features)
                    X_sub = X_round

                tree = XGBTree(max_depth=self.max_depth, lam=self.lam, gamma=self.gamma)
                tree.fit(X_sub, g[idx if self.subsample < 1.0 else slice(None), k], h[idx if self.subsample < 1.0 else slice(None), k])
                trees_this_round.append((tree, feat_mask))

                # Updating predictions
                y_pred[:, k] += self.learning_rate * tree.predict(X[:, feat_mask])

            self.trees.append(trees_this_round)

    def predict_proba(self, X):
        X = np.array(X)
        n_samples, _ = X.shape
        K = self.n_classes
        y_pred = np.zeros((n_samples, K))
        for trees_this_round in self.trees:
            for k, (tree, feat_mask) in enumerate(trees_this_round):
                y_pred[:, k] += self.learning_rate * tree.predict(X[:, feat_mask])
        return self._softmax(y_pred)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
# ----------------- Constructing the Random Forest Classifier -----------------
# Decision Tree Node
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold          # Threshold value to split
        self.left = left                    # Left subtree
        self.right = right                  # Right subtree
        self.value = value                  # Class label for leaf nodes

    def is_leaf_node(self):
        # Returns True if this node hold a value
        return self.value is not None

# Decision Tree Classifier
class DecisionTreeClassifier: 
    def __init__(self, max_depth = 10, min_samples_split = 2, feature_indices = None, n_classes = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_indices = feature_indices
        self.n_classes = n_classes
        self.n_thresholds = 15 

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        if self.feature_indices is None:
            self.feature_indices = np.arange(X.shape[1])
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, _ = X.shape
        
        # Base case 1: No samples
        if num_samples == 0:
            return None
        
        # Check purity
        num_unique_classes = len(np.unique(y))

        # Base case 2: Stopping conditions
        if (depth >= self.max_depth or num_unique_classes == 1 or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)
            
        # Greedy search for best split
        best_feat, best_thresh = self._best_split(X, y)
        
        # Base case 3: best split not found 
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        # Recursive case: Split is found
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        # If one of the parts is empty
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)
            
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return DecisionTreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        best_gain = -1 # Start with -1 to ensure first valid split is chosen
        split_idx, split_thresh = None, None
        
        parent_gini = self._gini(y)
        if parent_gini == 0: # Already pure
            return None, None

        for feat_idx in self.feature_indices:
            feature_column = X[:, feat_idx]
            unique_values = np.unique(feature_column)
            
            if len(unique_values) > self.n_thresholds:
                # Sample thresholds from unique values
                thresholds = np.random.choice(unique_values, self.n_thresholds, replace=False)
            else:
                thresholds = unique_values

            for thresh in thresholds:
                gain = self._gini_gain(y, feature_column, thresh, parent_gini)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh
                    
        return split_idx, split_thresh

    def _gini_gain(self, y, feature_column, threshold, parent_gini):
        # Generate splits
        left_idx = feature_column <= threshold
        right_idx = feature_column > threshold
        n = len(y)
        n_left, n_right = np.sum(left_idx), np.sum(right_idx)

        if n_left == 0 or n_right == 0:
            return 0 # No split was made

        # Weighted average Gini of children
        gini_left = self._gini(y[left_idx])
        gini_right = self._gini(y[right_idx])
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

        # Gini gain
        return parent_gini - child_gini

    def _gini(self, y):
        n = len(y)
        if n == 0:
            return 0
        counts = np.bincount(y, minlength = self.n_classes)
        probabilities = counts / n
        return 1.0 - np.sum(probabilities**2)

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0
        
        if self.n_classes:
            counts = np.bincount(y, minlength=self.n_classes)
        else:
            counts = np.bincount(y)
            
        return np.argmax(counts)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        # Base case
        if node is None: # Handle case where a branch is None
            return 0 # Default prediction
        if node.is_leaf_node():
            return node.value
        
        # Recursive calling of left and right branches
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)
        
#Random Forest Classifier
class RandomForest:
    def __init__(self, n_trees = 10, max_depth = 10, min_samples_split = 2, max_features = None, n_classes = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_classes = n_classes
        self.trees = []

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        _, n_features = X.shape
        
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
            
        self.trees = []
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        elif self.max_features > n_features:
            self.max_features = n_features
            
        print(f"  Training the Random Forest Classifier with n_trees: {self.n_trees}, max_depth: {self.max_depth}, max_features: {self.max_features}:")

        for i in range(self.n_trees):
            print(f'    Building tree {i+1}/{self.n_trees}...')
            
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Selecting feature indices
            all_feature_indices = np.arange(n_features)
            feature_indices = np.random.choice(all_feature_indices, self.max_features, replace = False)
            
            tree = DecisionTreeClassifier(max_depth = self.max_depth, min_samples_split = self.min_samples_split, feature_indices = feature_indices, n_classes = self.n_classes)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def predict_proba(self, X):
            X = np.array(X)
            n_samples = X.shape[0]
            tree_preds = np.array([tree.predict(X) for tree in self.trees])
            tree_preds_transposed = tree_preds.T
            probas = np.zeros((n_samples, self.n_classes))
            for i in range(n_samples):
                labels, counts = np.unique(tree_preds_transposed[i], return_counts=True)
                probas[i, labels] = counts / self.n_trees
            return probas
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
# ----------------- Constructing the k-Nearest Neighbour (KNN) Classifier -----------------
class KNNClassifier:
    def __init__(self, k = 5):
        self.k = k
        self.n_classes = None
        self.tree = None
        self.y_train = None

    def fit(self, X, y):
        print("  Note: this may take some time! (~30 seconds)")
        self.y_train = np.array(y)
        self.n_classes = len(np.unique(y))
        self.tree = spatial.cKDTree(X)

    def predict_proba(self, X):
        if self.tree is None:
            raise ValueError("Model not fitted yet.")
        
        _, indices = self.tree.query(X, k=self.k)

        if self.k == 1:
            indices = np.expand_dims(indices, axis=1)
        
        neighbor_labels = self.y_train[indices]
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes))
        
        for i in range(n_samples):
            labels, counts = np.unique(neighbor_labels[i], return_counts=True)
            probas[i, labels] = counts / self.k
            
        return probas

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)