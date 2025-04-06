import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from collections import Counter


class RandomWalkLinkPrediction:
    """
    Link prediction using random walk methods
    """
    def __init__(self, G, num_walks=10, walk_length=80, restart_prob=0.2):
        """
        Initialize the random walk link predictor
        
        Parameters:
        -----------
        G : networkx.Graph
            The input graph
        num_walks : int
            Number of random walks per node
        walk_length : int
            Length of each random walk
        restart_prob : float
            Probability of restarting the walk at the starting node (for PPR-like walks)
        """
        self.G = G
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.restart_prob = restart_prob
        self.node_visits = {}  # Dictionary to store visit frequencies
        
    def random_walk(self, start_node, walk_length, restart_prob=0.0):
        """
        Perform a single random walk from the start node
        
        Parameters:
        -----------
        start_node : node
            The starting node for the walk
        walk_length : int
            Length of the random walk
        restart_prob : float
            Probability of restarting the walk at the starting node
            
        Returns:
        --------
        walk : list
            List of nodes visited during the walk
        """
        walk = [start_node]
        current_node = start_node
        
        for _ in range(walk_length):
            # Get neighbors of current node
            neighbors = list(self.G.neighbors(current_node))
            
            # If no neighbors, restart the walk
            if not neighbors:
                current_node = start_node
                walk.append(current_node)
                continue
                
            # Check if we should restart based on restart probability
            if random.random() < restart_prob:
                current_node = start_node
            else:
                # Choose a random neighbor
                current_node = random.choice(neighbors)
                
            walk.append(current_node)
            
        return walk
        
    def perform_walks(self):
        """
        Perform random walks from each node and collect visit statistics
        """
        # Dictionary to store visit counts for each node pair
        self.node_visits = {}
        
        # Perform walks from each node
        nodes = list(self.G.nodes())
        
        for node in nodes:
            # Initialize visit counts for this starting node
            if node not in self.node_visits:
                self.node_visits[node] = Counter()
                
            # Perform multiple walks from this node
            for _ in range(self.num_walks):
                walk = self.random_walk(node, self.walk_length, self.restart_prob)
                
                # Count visits to other nodes during this walk (excluding self-visits)
                for visited_node in walk:
                    if visited_node != node:
                        self.node_visits[node][visited_node] += 1
        
        return self.node_visits
        
    def compute_proximity_scores(self):
        """
        Compute proximity scores between all node pairs based on random walks
        
        Returns:
        --------
        scores_dict : dict
            Dictionary of dictionaries with proximity scores between node pairs
        """
        # Make sure walks have been performed
        if not self.node_visits:
            self.perform_walks()
            
        # Create a normalized scores dictionary
        scores_dict = {}
        
        # Normalize by the number of walks and walk length
        normalization_factor = self.num_walks * self.walk_length
        
        for source, visits in self.node_visits.items():
            scores_dict[source] = {}
            for target, count in visits.items():
                scores_dict[source][target] = count / normalization_factor
                
        return scores_dict
        
    def predict_links(self, node_pairs):
        """
        Predict link scores for given node pairs
        
        Parameters:
        -----------
        node_pairs : list of tuples
            List of (source, target) node pairs to score
            
        Returns:
        --------
        scores : list
            List of scores for the provided node pairs
        """
        # Compute proximity scores if not already done
        if not hasattr(self, 'scores_dict'):
            self.scores_dict = self.compute_proximity_scores()
            
        scores = []
        
        for source, target in node_pairs:
            # Try to get score from source to target
            if source in self.scores_dict and target in self.scores_dict[source]:
                score = self.scores_dict[source][target]
            else:
                # If no directed walk, try reverse direction (for undirected considerations)
                if target in self.scores_dict and source in self.scores_dict[target]:
                    score = self.scores_dict[target][source]
                else:
                    # No path found in either direction
                    score = 0.0
                    
            scores.append(score)
            
        return scores
        
    def evaluate(self, positive_edges, negative_edges):
        """
        Evaluate the link prediction performance using AUC and AP
        
        Parameters:
        -----------
        positive_edges : list of tuples
            List of (source, target) node pairs that should be connected
        negative_edges : list of tuples
            List of (source, target) node pairs that should not be connected
            
        Returns:
        --------
        auc : float
            Area Under the ROC Curve
        ap : float
            Average Precision
        """
        # Predict scores
        pos_scores = self.predict_links(positive_edges)
        neg_scores = self.predict_links(negative_edges)
        
        # Combine scores and labels
        scores = pos_scores + neg_scores
        labels = [1] * len(pos_scores) + [0] * len(neg_scores)
        
        # Calculate metrics
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        
        return auc, ap