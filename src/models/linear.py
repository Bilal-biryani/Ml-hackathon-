"""Linear and probabilistic candidate scorers: Logistic Regression and Gaussian Naive Bayes."""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression():
    """L2-regularised logistic regression on standardised features."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def make_gaussian_nb():
    """Gaussian Naive Bayes (features treated as independent Gaussians per class)."""
    return GaussianNB()
