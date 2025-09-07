"""
Performance tests for weakspot analysis functionality.
Tests scalability, memory usage, and execution time.
"""

import pytest
import time
import psutil
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression

from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.weakspot_analyzer import WeakspotAnalyzer
from tests.test_fixtures.weakspot_test_data import (
    PerformanceTestScenarios,
    ExplainerFixtures
)


class TestWeakspotPerformance:
    """Performance tests for weakspot analysis."""
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def time_execution(self, func, *args, **kwargs):
        """Time function execution and return result and duration."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        return result, duration
    
    @pytest.mark.performance
    def test_large_dataset_performance(self):
        """Test performance with large dataset (10k samples)."""
        # Create large dataset
        X, y = PerformanceTestScenarios.create_large_dataset_scenario()
        
        # Create explainer
        explainer = ExplainerFixtures.create_classifier_explainer(X, y, 'rf')
        
        # Measure memory before
        memory_before = self.get_memory_usage()
        
        # Time the weakspot analysis
        result, duration = self.time_execution(
            explainer.calculate_weakspot_analysis,
            slice_features=['feature_0'],
            bins=10,
            min_samples=50
        )
        
        # Measure memory after
        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before
        
        # Performance assertions
        assert duration < 30.0, f"Analysis took too long: {duration:.2f}s"
        assert memory_used < 500, f"Used too much memory: {memory_used:.2f}MB"
        assert result is not None
        assert len(result['bin_results']) > 0
        
        print(f"Large dataset performance: {duration:.2f}s, {memory_used:.2f}MB")
    
    @pytest.mark.performance
    def test_high_dimensional_performance(self):
        """Test performance with high-dimensional dataset (100 features)."""
        X, y = PerformanceTestScenarios.create_high_dimensional_scenario()
        
        # Create explainer
        explainer = ExplainerFixtures.create_classifier_explainer(X, y, 'rf')
        
        # Time the analysis
        result, duration = self.time_execution(
            explainer.calculate_weakspot_analysis,
            slice_features=['feature_0', 'feature_1'],
            bins=5,
            min_samples=20
        )
        
        # Should handle high dimensions efficiently
        assert duration < 15.0, f"High-dimensional analysis took too long: {duration:.2f}s"
        assert result is not None
        
        print(f"High-dimensional performance: {duration:.2f}s")
    
    @pytest.mark.performance
    def test_many_bins_performance(self):
        """Test performance with many bins."""
        X, y = PerformanceTestScenarios.create_many_bins_scenario()
        
        # Create explainer
        explainer = ExplainerFixtures.create_classifier_explainer(X, y, 'rf')
        
        # Test with maximum allowed bins
        result, duration = self.time_execution(
            explainer.calculate_weakspot_analysis,
            slice_features=['feature_0'],
            bins=50,  # Maximum allowed
            min_samples=10
        )
        
        # Should handle many bins efficiently
        assert duration < 10.0, f"Many bins analysis took too long: {duration:.2f}s"
        assert result is not None
        assert len(result['bin_results']) <= 50
        
        print(f"Many bins performance: {duration:.2f}s")
    
    @pytest.mark.performance
    def test_2d_analysis_performance(self):
        """Test performance of 2D analysis (feature interactions)."""
        X, y = PerformanceTestScenarios.create_large_dataset_scenario()
        
        # Create explainer
        explainer = ExplainerFixtures.create_classifier_explainer(X, y, 'rf')
        
        # Time 2D analysis
        result, duration = self.time_execution(
            explainer.calculate_weakspot_analysis,
            slice_features=['feature_0', 'feature_1'],
            bins=10,  # 10x10 = 100 slices
            min_samples=20
        )
        
        # 2D analysis should still be reasonable
        assert duration < 20.0, f"2D analysis took too long: {duration:.2f}s"
        assert result is not None
        
        print(f"2D analysis performance: {duration:.2f}s")
    
    @pytest.mark.performance
    def test_tree_slicing_performance(self):
        """Test performance of tree-based slicing."""
        X, y = PerformanceTestScenarios.create_large_dataset_scenario()
        
        # Create explainer
        explainer = ExplainerFixtures.create_classifier_explainer(X, y, 'rf')
        
        # Time tree slicing
        result, duration = self.time_execution(
            explainer.calculate_weakspot_analysis,
            slice_features=['feature_0', 'feature_1'],
            slice_method='tree',
            min_samples=50
        )
        
        # Tree slicing should be efficient
        assert duration < 15.0, f"Tree slicing took too long: {duration:.2f}s"
        assert result is not None
        
        print(f"Tree slicing performance: {duration:.2f}s")
    
    @pytest.mark.performance
    def test_regression_performance(self):
        """Test performance with regression models."""
        # Create large regression dataset
        X, y = make_regression(
            n_samples=8000,
            n_features=15,
            n_informative=10,
            noise=0.1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        # Create explainer
        explainer = ExplainerFixtures.create_regression_explainer(X_df, y_series, 'rf')
        
        # Time regression analysis
        result, duration = self.time_execution(
            explainer.calculate_weakspot_analysis,
            slice_features=['feature_0'],
            bins=15,
            min_samples=30
        )
        
        # Regression should be as fast as classification
        assert duration < 25.0, f"Regression analysis took too long: {duration:.2f}s"
        assert result is not None
        
        print(f"Regression performance: {duration:.2f}s")
    
    @pytest.mark.performance
    def test_multiple_metrics_performance(self):
        """Test performance when calculating multiple metrics."""
        X, y = PerformanceTestScenarios.create_large_dataset_scenario()
        
        # Create explainer
        explainer = ExplainerFixtures.create_classifier_explainer(X, y, 'rf')
        
        # Test different metrics
        metrics = ['accuracy', 'log_loss', 'brier_score']
        total_duration = 0
        
        for metric in metrics:
            result, duration = self.time_execution(
                explainer.calculate_weakspot_analysis,
                slice_features=['feature_0'],
                metric=metric,
                bins=10,
                min_samples=30
            )
            total_duration += duration
            assert result is not None
        
        # Multiple metrics should not be prohibitively slow
        assert total_duration < 45.0, f"Multiple metrics took too long: {total_duration:.2f}s"
        
        print(f"Multiple metrics performance: {total_duration:.2f}s")
    
    @pytest.mark.performance
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Create very large dataset
        X, y = make_classification(
            n_samples=15000,
            n_features=25,
            n_informative=20,
            n_redundant=3,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        # Create explainer
        explainer = ExplainerFixtures.create_classifier_explainer(X_df, y_series, 'rf')
        
        # Measure memory usage
        memory_before = self.get_memory_usage()
        
        # Run analysis
        result = explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            bins=20,
            min_samples=50
        )
        
        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before
        
        # Should not use excessive memory
        assert memory_used < 1000, f"Used too much memory: {memory_used:.2f}MB"
        assert result is not None
        
        print(f"Memory efficiency: {memory_used:.2f}MB for 15k samples")
    
    @pytest.mark.performance
    def test_concurrent_analysis_performance(self):
        """Test performance when running multiple analyses concurrently."""
        import threading
        import queue
        
        X, y = PerformanceTestScenarios.create_large_dataset_scenario()
        explainer = ExplainerFixtures.create_classifier_explainer(X, y, 'rf')
        
        results_queue = queue.Queue()
        
        def run_analysis(feature_name):
            """Run analysis for a specific feature."""
            try:
                start_time = time.time()
                result = explainer.calculate_weakspot_analysis(
                    slice_features=[feature_name],
                    bins=8,
                    min_samples=40
                )
                duration = time.time() - start_time
                results_queue.put((feature_name, result, duration, None))
            except Exception as e:
                results_queue.put((feature_name, None, 0, e))
        
        # Start multiple threads
        threads = []
        features_to_test = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        
        start_time = time.time()
        
        for feature in features_to_test:
            thread = threading.Thread(target=run_analysis, args=(feature,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all analyses completed successfully
        assert len(results) == len(features_to_test)
        for feature, result, duration, error in results:
            assert error is None, f"Analysis failed for {feature}: {error}"
            assert result is not None
            assert duration > 0
        
        # Concurrent execution should be faster than sequential
        assert total_time < 60.0, f"Concurrent analysis took too long: {total_time:.2f}s"
        
        print(f"Concurrent analysis performance: {total_time:.2f}s for {len(features_to_test)} features")


class TestWeakspotScalability:
    """Scalability tests for different dataset sizes."""
    
    @pytest.mark.performance
    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 2000, 5000])
    def test_scalability_by_sample_size(self, n_samples):
        """Test how performance scales with sample size."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        explainer = ExplainerFixtures.create_classifier_explainer(X_df, y_series, 'rf')
        
        start_time = time.time()
        result = explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            bins=10,
            min_samples=max(5, n_samples // 100)  # Adaptive min_samples
        )
        duration = time.time() - start_time
        
        # Performance should scale reasonably
        expected_max_time = max(5.0, n_samples / 1000 * 10)  # Linear scaling expectation
        assert duration < expected_max_time, f"Analysis for {n_samples} samples took {duration:.2f}s"
        assert result is not None
        
        print(f"Scalability test: {n_samples} samples in {duration:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.parametrize("n_features", [5, 10, 20, 50])
    def test_scalability_by_feature_count(self, n_features):
        """Test how performance scales with feature count."""
        X, y = make_classification(
            n_samples=2000,
            n_features=n_features,
            n_informative=min(n_features, max(3, n_features // 2)),
            n_redundant=min(n_features // 4, 5),
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        explainer = ExplainerFixtures.create_classifier_explainer(X_df, y_series, 'rf')
        
        start_time = time.time()
        result = explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            bins=10,
            min_samples=30
        )
        duration = time.time() - start_time
        
        # Feature count should not significantly impact performance for single feature analysis
        assert duration < 15.0, f"Analysis with {n_features} features took {duration:.2f}s"
        assert result is not None
        
        print(f"Feature scalability test: {n_features} features in {duration:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.parametrize("n_bins", [5, 10, 20, 30, 50])
    def test_scalability_by_bin_count(self, n_bins):
        """Test how performance scales with bin count."""
        X, y = PerformanceTestScenarios.create_many_bins_scenario()
        
        explainer = ExplainerFixtures.create_classifier_explainer(X, y, 'rf')
        
        start_time = time.time()
        result = explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            bins=n_bins,
            min_samples=20
        )
        duration = time.time() - start_time
        
        # Performance should scale reasonably with bin count
        expected_max_time = max(3.0, n_bins / 10 * 5)  # Roughly linear scaling
        assert duration < expected_max_time, f"Analysis with {n_bins} bins took {duration:.2f}s"
        assert result is not None
        
        print(f"Bin scalability test: {n_bins} bins in {duration:.2f}s")


class TestWeakspotMemoryUsage:
    """Memory usage tests for weakspot analysis."""
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.performance
    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after analysis."""
        initial_memory = self.get_memory_usage()
        
        # Run multiple analyses
        for i in range(5):
            X, y = make_classification(
                n_samples=2000,
                n_features=10,
                random_state=42 + i
            )
            
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
            y_series = pd.Series(y, name='target')
            
            explainer = ExplainerFixtures.create_classifier_explainer(X_df, y_series, 'rf')
            
            result = explainer.calculate_weakspot_analysis(
                slice_features=['feature_0'],
                bins=10,
                min_samples=30
            )
            
            assert result is not None
            
            # Force garbage collection
            import gc
            gc.collect()
        
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 200, f"Memory increased by {memory_increase:.2f}MB"
        
        print(f"Memory cleanup test: {memory_increase:.2f}MB increase after 5 analyses")
    
    @pytest.mark.performance
    def test_memory_with_large_results(self):
        """Test memory usage when analysis produces large results."""
        # Create dataset that will produce many slices
        X, y = make_classification(
            n_samples=10000,
            n_features=8,
            n_informative=6,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        explainer = ExplainerFixtures.create_classifier_explainer(X_df, y_series, 'rf')
        
        memory_before = self.get_memory_usage()
        
        # Run 2D analysis with many bins (will create many slices)
        result = explainer.calculate_weakspot_analysis(
            slice_features=['feature_0', 'feature_1'],
            bins=20,  # 20x20 = 400 potential slices
            min_samples=10  # Low threshold to keep more slices
        )
        
        memory_after = self.get_memory_usage()
        memory_used = memory_after - memory_before
        
        # Even with large results, memory usage should be reasonable
        assert memory_used < 300, f"Large results used too much memory: {memory_used:.2f}MB"
        assert result is not None
        assert len(result['bin_results']) > 0
        
        print(f"Large results memory test: {memory_used:.2f}MB for {len(result['bin_results'])} slices")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])