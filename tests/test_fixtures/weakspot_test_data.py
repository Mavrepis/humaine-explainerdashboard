"""
Test data fixtures for weakspot analysis testing.
Provides various datasets and scenarios for comprehensive testing.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.datasets import titanic_survive, titanic_fare


class WeakspotTestFixtures:
    """Collection of test fixtures for weakspot analysis testing."""
    
    @staticmethod
    def create_small_classification_dataset():
        """Create small classification dataset for edge case testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=3,
            n_informative=2,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_large_classification_dataset():
        """Create large classification dataset for performance testing."""
        X, y = make_classification(
            n_samples=5000,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_clusters_per_class=2,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_imbalanced_classification_dataset():
        """Create highly imbalanced classification dataset."""
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            weights=[0.9, 0.1],  # Highly imbalanced
            flip_y=0.01,  # Add some noise
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_multiclass_dataset():
        """Create multiclass classification dataset."""
        X, y = make_classification(
            n_samples=1000,
            n_features=6,
            n_informative=4,
            n_redundant=1,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_regression_with_outliers():
        """Create regression dataset with outliers."""
        X, y = make_regression(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42
        )
        
        # Add outliers to create weak regions
        outlier_indices = np.random.choice(len(y), size=50, replace=False)
        y[outlier_indices] += np.random.normal(0, 10, size=50)
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_nonlinear_regression_dataset():
        """Create regression dataset with nonlinear relationships."""
        n_samples = 1000
        X = np.random.uniform(-3, 3, (n_samples, 4))
        
        # Create nonlinear target with different patterns in different regions
        y = (
            X[:, 0] ** 2 +  # Quadratic relationship
            np.sin(X[:, 1] * 2) * 3 +  # Sinusoidal relationship
            np.where(X[:, 2] > 0, X[:, 2] * 2, X[:, 2] * 0.5) +  # Piecewise linear
            X[:, 3] +  # Linear relationship
            np.random.normal(0, 0.5, n_samples)  # Noise
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_categorical_features_dataset():
        """Create dataset with mixed categorical and numerical features."""
        n_samples = 1000
        
        # Numerical features
        numerical_features = np.random.randn(n_samples, 3)
        
        # Categorical features
        categorical_feature_1 = np.random.choice(['A', 'B', 'C'], size=n_samples)
        categorical_feature_2 = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
        
        # Create target with interactions
        y = (
            numerical_features[:, 0] + 
            numerical_features[:, 1] * 0.5 +
            (categorical_feature_1 == 'A').astype(int) * 2 +
            (categorical_feature_2 == 'X').astype(int) * 1.5 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Convert to binary classification
        y = (y > np.median(y)).astype(int)
        
        X_df = pd.DataFrame({
            'num_feature_0': numerical_features[:, 0],
            'num_feature_1': numerical_features[:, 1],
            'num_feature_2': numerical_features[:, 2],
            'cat_feature_1': categorical_feature_1,
            'cat_feature_2': categorical_feature_2
        })
        
        # One-hot encode for model training
        X_encoded = pd.get_dummies(X_df, columns=['cat_feature_1', 'cat_feature_2'])
        y_series = pd.Series(y, name='target')
        
        return X_encoded, y_series
    
    @staticmethod
    def create_dataset_with_missing_values():
        """Create dataset with missing values to test robustness."""
        X, y = make_classification(
            n_samples=500,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        # Introduce missing values randomly
        missing_mask = np.random.random(X_df.shape) < 0.05  # 5% missing
        X_df = X_df.mask(missing_mask)
        
        return X_df, y_series
    
    @staticmethod
    def create_perfect_separation_dataset():
        """Create dataset with perfect separation to test edge cases."""
        n_samples = 500
        X = np.random.randn(n_samples, 3)
        
        # Create perfect separation based on first feature
        y = (X[:, 0] > 0).astype(int)
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_constant_target_dataset():
        """Create dataset with constant target to test edge cases."""
        X, _ = make_classification(
            n_samples=300,
            n_features=4,
            n_informative=2,
            n_redundant=1,
            random_state=42
        )
        
        # Make all targets the same
        y = np.ones(len(X))
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series


@pytest.fixture(scope="session")
def small_classification_data():
    """Small classification dataset fixture."""
    return WeakspotTestFixtures.create_small_classification_dataset()


@pytest.fixture(scope="session")
def large_classification_data():
    """Large classification dataset fixture."""
    return WeakspotTestFixtures.create_large_classification_dataset()


@pytest.fixture(scope="session")
def imbalanced_classification_data():
    """Imbalanced classification dataset fixture."""
    return WeakspotTestFixtures.create_imbalanced_classification_dataset()


@pytest.fixture(scope="session")
def multiclass_data():
    """Multiclass classification dataset fixture."""
    return WeakspotTestFixtures.create_multiclass_dataset()


@pytest.fixture(scope="session")
def regression_with_outliers_data():
    """Regression dataset with outliers fixture."""
    return WeakspotTestFixtures.create_regression_with_outliers()


@pytest.fixture(scope="session")
def nonlinear_regression_data():
    """Nonlinear regression dataset fixture."""
    return WeakspotTestFixtures.create_nonlinear_regression_dataset()


@pytest.fixture(scope="session")
def categorical_features_data():
    """Dataset with categorical features fixture."""
    return WeakspotTestFixtures.create_categorical_features_dataset()


@pytest.fixture(scope="session")
def missing_values_data():
    """Dataset with missing values fixture."""
    return WeakspotTestFixtures.create_dataset_with_missing_values()


@pytest.fixture(scope="session")
def perfect_separation_data():
    """Dataset with perfect separation fixture."""
    return WeakspotTestFixtures.create_perfect_separation_dataset()


@pytest.fixture(scope="session")
def constant_target_data():
    """Dataset with constant target fixture."""
    return WeakspotTestFixtures.create_constant_target_dataset()


@pytest.fixture(scope="session")
def titanic_classification_data():
    """Real-world Titanic classification dataset."""
    X_train, y_train, X_test, y_test = titanic_survive()
    # Combine for full dataset
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    return X_full, y_full


@pytest.fixture(scope="session")
def titanic_regression_data():
    """Real-world Titanic regression dataset."""
    X_train, y_train, X_test, y_test = titanic_fare()
    # Combine for full dataset
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    return X_full, y_full


class ModelFixtures:
    """Collection of trained model fixtures for testing."""
    
    @staticmethod
    def create_rf_classifier(X, y):
        """Create trained Random Forest classifier."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @staticmethod
    def create_rf_regressor(X, y):
        """Create trained Random Forest regressor."""
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @staticmethod
    def create_logistic_regression(X, y):
        """Create trained Logistic Regression classifier."""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model
    
    @staticmethod
    def create_linear_regression(X, y):
        """Create trained Linear Regression model."""
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    @staticmethod
    def create_decision_tree_classifier(X, y):
        """Create trained Decision Tree classifier."""
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
        model.fit(X, y)
        return model
    
    @staticmethod
    def create_decision_tree_regressor(X, y):
        """Create trained Decision Tree regressor."""
        model = DecisionTreeRegressor(random_state=42, max_depth=5)
        model.fit(X, y)
        return model


class ExplainerFixtures:
    """Collection of explainer fixtures for testing."""
    
    @staticmethod
    def create_classifier_explainer(X, y, model_type='rf'):
        """Create ClassifierExplainer with specified model type."""
        if model_type == 'rf':
            model = ModelFixtures.create_rf_classifier(X, y)
        elif model_type == 'logistic':
            model = ModelFixtures.create_logistic_regression(X, y)
        elif model_type == 'dt':
            model = ModelFixtures.create_decision_tree_classifier(X, y)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        explainer = ClassifierExplainer(model, X, y)
        return explainer
    
    @staticmethod
    def create_regression_explainer(X, y, model_type='rf'):
        """Create RegressionExplainer with specified model type."""
        if model_type == 'rf':
            model = ModelFixtures.create_rf_regressor(X, y)
        elif model_type == 'linear':
            model = ModelFixtures.create_linear_regression(X, y)
        elif model_type == 'dt':
            model = ModelFixtures.create_decision_tree_regressor(X, y)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        explainer = RegressionExplainer(model, X, y)
        return explainer


# Performance test scenarios
class PerformanceTestScenarios:
    """Scenarios for performance testing."""
    
    @staticmethod
    def create_large_dataset_scenario():
        """Create large dataset for performance testing."""
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_clusters_per_class=3,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_high_dimensional_scenario():
        """Create high-dimensional dataset for testing."""
        X, y = make_classification(
            n_samples=1000,
            n_features=100,
            n_informative=50,
            n_redundant=20,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @staticmethod
    def create_many_bins_scenario():
        """Create scenario for testing with many bins."""
        X, y = make_classification(
            n_samples=5000,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series


@pytest.fixture(scope="session")
def large_dataset_scenario():
    """Large dataset performance test scenario."""
    return PerformanceTestScenarios.create_large_dataset_scenario()


@pytest.fixture(scope="session")
def high_dimensional_scenario():
    """High-dimensional dataset performance test scenario."""
    return PerformanceTestScenarios.create_high_dimensional_scenario()


@pytest.fixture(scope="session")
def many_bins_scenario():
    """Many bins performance test scenario."""
    return PerformanceTestScenarios.create_many_bins_scenario()