"""
Five probability sampling techniques for creating balanced datasets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class SamplingTechniques:
    def __init__(self, X, y, random_state=42):
        """Initialize with feature matrix and labels"""
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.random_state = random_state
        np.random.seed(random_state)
    
    def simple_random_sampling(self, sample_size=None):
        """
        Simple Random Sampling: Every element has equal probability
        Returns random subset without replacement
        """
        if sample_size is None:
            sample_size = int(0.7 * self.n_samples)
        
        indices = np.random.choice(
            self.n_samples, size=sample_size, replace=False
        )
        return self.X[indices], self.y[indices], indices
    
    def stratified_random_sampling(self, sample_size=None):
        """
        Stratified Random Sampling: Maintains class proportions
        Each class is sampled proportionally
        """
        if sample_size is None:
            sample_size = int(0.7 * self.n_samples)
        
        # Get unique classes and their indices
        unique_classes = np.unique(self.y)
        indices = []
        
        for class_label in unique_classes:
            class_indices = np.where(self.y == class_label)[0]
            class_sample_size = int(sample_size * len(class_indices) / self.n_samples)
            
            sampled_indices = np.random.choice(
                class_indices, size=class_sample_size, replace=False
            )
            indices.extend(sampled_indices)
        
        indices = np.array(indices)
        return self.X[indices], self.y[indices], indices
    
    def cluster_sampling(self, n_clusters=5):
        """
        Cluster Sampling: Divide population into clusters, randomly sample clusters
        Then select samples from selected clusters
        """
        # Divide data into clusters (by value ranges)
        cluster_size = self.n_samples // n_clusters
        clusters = [np.arange(i * cluster_size, (i + 1) * cluster_size) 
                   for i in range(n_clusters)]
        
        # Randomly select clusters (50% of clusters)
        selected_clusters = np.random.choice(
            n_clusters, size=n_clusters // 2, replace=False
        )
        
        # Collect indices from selected clusters
        indices = []
        for cluster_idx in selected_clusters:
            cluster_indices = clusters[cluster_idx]
            # Sample 70% from each selected cluster
            sample_indices = np.random.choice(
                cluster_indices,
                size=len(cluster_indices) * 7 // 10,
                replace=False
            )
            indices.extend(sample_indices)
        
        indices = np.array(indices)
        return self.X[indices], self.y[indices], indices
    
    def systematic_sampling(self, k=3):
        """
        Systematic Sampling: Select every kth element
        First element is random, then every kth element is selected
        """
        # Random starting point
        start = np.random.randint(0, k)
        
        # Every kth element
        indices = np.arange(start, self.n_samples, k)
        
        return self.X[indices], self.y[indices], indices
    
    def kfold_sampling(self, n_splits=5):
        """
        K-Fold Cross-Validation Sampling: Divide into k folds
        Returns list of (train_indices, test_indices) tuples
        """
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        folds = []
        
        for train_indices, test_indices in kfold.split(self.X):
            folds.append({
                'train': (self.X[train_indices], self.y[train_indices]),
                'test': (self.X[test_indices], self.y[test_indices]),
                'train_idx': train_indices,
                'test_idx': test_indices
            })
        
        return folds
    
    def get_all_samples(self):
        """Get all 5 different samples"""
        samples = {
            'Sampling1_SimpleRandom': self.simple_random_sampling(),
            'Sampling2_Stratified': self.stratified_random_sampling(),
            'Sampling3_Cluster': self.cluster_sampling(),
            'Sampling4_Systematic': self.systematic_sampling(),
            'Sampling5_KFold': self.kfold_sampling()
        }
        return samples


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, weights=[0.9, 0.1], random_state=42
    )
    
    sampler = SamplingTechniques(X, y)
    
    print("[*] Testing Sampling Techniques")
    X_sample, y_sample, indices = sampler.simple_random_sampling()
    print(f"Simple Random: {X_sample.shape}, {y_sample.shape}")
    
    X_sample, y_sample, indices = sampler.stratified_random_sampling()
    print(f"Stratified: {X_sample.shape}, {y_sample.shape}")
    
    X_sample, y_sample, indices = sampler.cluster_sampling()
    print(f"Cluster: {X_sample.shape}, {y_sample.shape}")
    
    X_sample, y_sample, indices = sampler.systematic_sampling()
    print(f"Systematic: {X_sample.shape}, {y_sample.shape}")
    
    folds = sampler.kfold_sampling()
    print(f"K-Fold: {len(folds)} folds")