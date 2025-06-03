import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def k_fold_score(X, labels, indices=None, k=5, precision=10, random_state=0):
    """
    Performs k-fold cross-validation and returns the average weighted F1 score
    """
    if indices is None:
        indices = np.arange(X.shape[0])

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = []

    for train_idx, test_idx in kf.split(indices):
        train_idx = indices[train_idx]
        test_idx = indices[test_idx]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = LogisticRegression(C=10, penalty="l2", max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = f1_score(y_test, y_pred, average='weighted')
        scores.append(score)

    return round(np.mean(scores), precision)