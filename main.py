from algorithms import * 
import time
from sklearn.metrics import accuracy_score, f1_score

 # ----------------- Ensembling the classification models described in algorithms.py -----------------
class WeightedEnsemble:
    def __init__(self, model_configs, weights):
        self.model_configs = model_configs
        self.weights = weights
        self.models = {}
        self.n_classes = 10

        total_weight = sum(weights.values())
        self.norm_weights = {name: w / total_weight for name, w in weights.items()}

    def fit(self, X, y):
        print("Beginning full ensemble training using Softmax Regression, XGBoost Classifier, Random Forest Classifier & KNN:")
        print("The ensemble weights are distributed as:")
        for name, w in self.norm_weights.items():
            print(f"  {name}: {w*100:.2f}%")
        start_time = time.time()
        
        for name, (ModelClass, params) in self.model_configs.items():
            print(f"\nTraining model {name}:")
            model_start_time = time.time()
            model = ModelClass(**params)
            model.fit(X, y)
            self.models[name] = model
            print(f"Training time for {name}: {time.time() - model_start_time:}s")
            
        print(f"Full ensemble training complete in: {time.time() - start_time:.2f}s!")

    def _get_all_probas(self, X):
        probas_dict = {}
        for name, model in self.models.items():
            probas_dict[name] = model.predict_proba(X)
        return probas_dict

    '''
    def _predict_proba_soft(self, X):
        # Soft Voting (weighted average of probabilities)
        all_probas = self._get_all_probas(X)
        weighted_probas = np.zeros((X.shape[0], self.n_classes))
        for name, probas in all_probas.items():
            weight = self.norm_weights[name]
            weighted_probas += probas * weight
        return weighted_probas

    def predict_soft(self, X):
        # Returns final class for soft voting
        return np.argmax(self._predict_proba_soft(X), axis=1)
    '''

    def predict_hard(self, X):
        # Hard Voting (weighted vote on final classes)
        n_samples = X.shape[0]
        all_preds = {}
        for name, model in self.models.items():
            all_preds[name] = model.predict(X)

        final_preds = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            votes = {}
            for name, preds in all_preds.items():
                pred_class = preds[i]
                weight = self.norm_weights[name]
                votes[pred_class] = votes.get(pred_class, 0.0) + weight
            final_preds[i] = max(votes, key=votes.get)   
        return final_preds
    
if __name__ == "__main__":
    Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')

    model_configurations = {
    'Softmax': (BaggingSoftmaxClassifier, {'n_estimators': 10, 'learning_rate': 0.05, 'n_epochs': 200, 'mini_batch_size': 256}),
    'XGBoost': (XGBoostMultiClassifier, {'n_classes': 10, 'n_estimators': 30, 'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.5, 'subsample': 0.8}),
    'RandomForest': (RandomForest, {'n_trees': 50, 'max_depth': 8, 'min_samples_split': 5, 'n_classes': 10, 'max_features': 40}),
    'KNN': (KNNClassifier, {'k': 5})}
        
    ensemble_weights = {'Softmax': 35, 'XGBoost': 10, 'RandomForest': 10, 'KNN': 45}
    
    ensemble = WeightedEnsemble(model_configurations, ensemble_weights)
    ensemble.fit(Xtrain, ytrain)
    print("Validating now on the training and validation data:")
    # Method 2: Hard Voting (Weighted Class Vote)
    print("\nValidating on training data via hard voting...")
    print("Note: this may take some time! (~100 seconds)")
    start_time = time.time()
    ytrain_pred_hard = ensemble.predict_hard(Xtrain)
    print(f"  Time taken: {time.time() - start_time}s")
    print(f"  Accuracy: {accuracy_score(ytrain, ytrain_pred_hard)}")
    print(f"  F1-Score: {f1_score(ytrain, ytrain_pred_hard, average="macro")}") 

    print("\nValidating on validation data via hard voting...")
    print("Note: this may take some time! (~20 seconds)")
    start_time = time.time()
    yval_pred_hard = ensemble.predict_hard(Xval)
    print(f"  Time taken: {time.time() - start_time}s")
    print(f"  Accuracy: {accuracy_score(yval, yval_pred_hard)}")
    print(f"  F1-Score: {f1_score(yval, yval_pred_hard, average="macro")}")   