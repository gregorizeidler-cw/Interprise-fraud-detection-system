"""
Graph Neural Networks for Advanced Fraud Detection

This module implements state-of-the-art graph neural network architectures
for detecting fraud rings, suspicious networks, and complex relationship patterns
in financial transaction data.

Key Features:
- Transaction graph construction and analysis
- GraphSAGE for scalable neighborhood sampling
- Temporal graph networks for time-aware embeddings
- Community detection and fraud ring identification
- Node2Vec embeddings for entity representation
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TransactionGraphBuilder:
    """
    Builds various types of graphs from transaction data for GNN analysis
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.graphs = {}
        
    def build_customer_transaction_graph(self, transactions_df: pd.DataFrame) -> nx.Graph:
        """
        Build a graph where customers are nodes and transactions are edges
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            NetworkX graph with customer nodes and transaction edges
        """
        logger.info("Building customer transaction graph...")
        
        G = nx.Graph()
        
        # Add customer nodes with features
        customers = set(transactions_df['customer_id'].unique())
        for customer in customers:
            customer_data = self._get_customer_features(customer, transactions_df)
            G.add_node(customer, **customer_data)
        
        # Add transaction edges with features
        for _, txn in transactions_df.iterrows():
            if 'beneficiary_id' in txn and pd.notna(txn['beneficiary_id']):
                customer_id = txn['customer_id']
                beneficiary_id = txn['beneficiary_id']
                
                # Add beneficiary as node if not exists
                if beneficiary_id not in G.nodes():
                    G.add_node(beneficiary_id, node_type='beneficiary')
                
                # Add/update edge with transaction features
                if G.has_edge(customer_id, beneficiary_id):
                    # Update existing edge
                    edge_data = G[customer_id][beneficiary_id]
                    edge_data['transaction_count'] += 1
                    edge_data['total_amount'] += txn['amount']
                    edge_data['last_transaction'] = txn['timestamp']
                else:
                    # Create new edge
                    G.add_edge(
                        customer_id, 
                        beneficiary_id,
                        transaction_count=1,
                        total_amount=txn['amount'],
                        first_transaction=txn['timestamp'],
                        last_transaction=txn['timestamp'],
                        product_types=[txn['product_type']]
                    )
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def build_device_sharing_graph(self, transactions_df: pd.DataFrame) -> nx.Graph:
        """
        Build a graph based on shared devices between customers
        """
        logger.info("Building device sharing graph...")
        
        G = nx.Graph()
        
        # Group by device_id to find customers sharing devices
        device_groups = transactions_df.groupby('device_id')['customer_id'].unique()
        
        for device_id, customers in device_groups.items():
            if len(customers) > 1:  # Device shared by multiple customers
                # Add edges between all customers sharing this device
                for i, customer1 in enumerate(customers):
                    for customer2 in customers[i+1:]:
                        if G.has_edge(customer1, customer2):
                            G[customer1][customer2]['shared_devices'] += 1
                        else:
                            G.add_edge(
                                customer1, 
                                customer2, 
                                shared_devices=1,
                                device_ids=[device_id]
                            )
        
        return G
    
    def _get_customer_features(self, customer_id: str, transactions_df: pd.DataFrame) -> Dict:
        """Get customer node features"""
        customer_txns = transactions_df[transactions_df['customer_id'] == customer_id]
        
        return {
            'node_type': 'customer',
            'transaction_count': len(customer_txns),
            'total_volume': customer_txns['amount'].sum(),
            'avg_amount': customer_txns['amount'].mean(),
            'unique_beneficiaries': customer_txns['beneficiary_id'].nunique(),
            'product_diversity': customer_txns['product_type'].nunique(),
            'days_active': (customer_txns['timestamp'].max() - customer_txns['timestamp'].min()).days
        }


class GraphSAGEFraudDetector(nn.Module):
    """
    GraphSAGE implementation for fraud detection
    
    This model learns node embeddings by sampling and aggregating features
    from a node's local neighborhood
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 output_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super(GraphSAGEFraudDetector, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for graph batching
        """
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Node embeddings
        node_embeddings = x
        
        # Classification
        if batch is not None:
            # Graph-level prediction (for batch processing)
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        
        logits = self.classifier(x)
        
        return {
            'logits': logits,
            'node_embeddings': node_embeddings,
            'probabilities': F.softmax(logits, dim=-1)
        }


class TemporalGraphAttention(nn.Module):
    """
    Temporal Graph Attention Network for time-aware fraud detection
    
    Incorporates temporal information into graph neural networks
    to capture evolving fraud patterns over time
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 temporal_dim: int = 32):
        super(TemporalGraphAttention, self).__init__()
        
        self.temporal_dim = temporal_dim
        
        # Temporal encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, temporal_dim),
            nn.ReLU(),
            nn.Linear(temporal_dim, temporal_dim)
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(input_dim + temporal_dim, hidden_dim, heads=num_heads, concat=True)
        )
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
            )
        
        # Final classification
        self.classifier = nn.Linear(hidden_dim * num_heads, 2)
        
    def forward(self, x, edge_index, edge_timestamps):
        """
        Forward pass with temporal information
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_timestamps: Timestamps for each edge
        """
        # Encode temporal information
        temporal_features = self.temporal_encoder(edge_timestamps.unsqueeze(-1))
        
        # Combine node features with temporal encoding
        # This is a simplified version - in practice, you'd want more sophisticated temporal fusion
        temporal_node_features = torch.cat([x, temporal_features[:x.size(0)]], dim=-1)
        
        # Graph attention layers
        for gat_layer in self.gat_layers:
            temporal_node_features = gat_layer(temporal_node_features, edge_index)
            temporal_node_features = F.elu(temporal_node_features)
        
        # Classification
        logits = self.classifier(temporal_node_features)
        
        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1)
        }


class CommunityFraudDetector:
    """
    Detects fraud rings and suspicious communities in transaction graphs
    """
    
    def __init__(self):
        self.communities = {}
        self.fraud_scores = {}
        
    def detect_communities(self, graph: nx.Graph, method: str = 'louvain') -> Dict:
        """
        Detect communities in the graph using various algorithms
        
        Args:
            graph: NetworkX graph
            method: Community detection method ('louvain', 'girvan_newman', 'label_propagation')
        """
        logger.info(f"Detecting communities using {method} method...")
        
        if method == 'louvain':
            import community as community_louvain
            communities = community_louvain.best_partition(graph)
            
        elif method == 'girvan_newman':
            from networkx.algorithms import community
            communities_iter = community.girvan_newman(graph)
            # Get first level of communities
            communities_list = next(communities_iter)
            communities = {}
            for i, community_nodes in enumerate(communities_list):
                for node in community_nodes:
                    communities[node] = i
                    
        elif method == 'label_propagation':
            from networkx.algorithms import community
            communities_list = community.label_propagation_communities(graph)
            communities = {}
            for i, community_nodes in enumerate(communities_list):
                for node in community_nodes:
                    communities[node] = i
        
        self.communities = communities
        return communities
    
    def score_communities_for_fraud(self, graph: nx.Graph, communities: Dict) -> Dict:
        """
        Score each community for fraud likelihood based on various metrics
        """
        community_scores = {}
        
        # Group nodes by community
        community_groups = {}
        for node, community_id in communities.items():
            if community_id not in community_groups:
                community_groups[community_id] = []
            community_groups[community_id].append(node)
        
        for community_id, nodes in community_groups.items():
            subgraph = graph.subgraph(nodes)
            
            # Calculate fraud indicators
            metrics = {
                'size': len(nodes),
                'density': nx.density(subgraph),
                'avg_clustering': np.mean(list(nx.clustering(subgraph).values())) if len(nodes) > 1 else 0,
                'avg_degree': np.mean([d for n, d in subgraph.degree()]) if len(nodes) > 0 else 0,
                'transaction_velocity': self._calculate_transaction_velocity(subgraph),
                'amount_concentration': self._calculate_amount_concentration(subgraph)
            }
            
            # Combine metrics into fraud score
            fraud_score = self._calculate_community_fraud_score(metrics)
            community_scores[community_id] = {
                'fraud_score': fraud_score,
                'metrics': metrics,
                'nodes': nodes
            }
        
        return community_scores
    
    def _calculate_transaction_velocity(self, subgraph: nx.Graph) -> float:
        """Calculate average transaction velocity in community"""
        velocities = []
        for u, v, data in subgraph.edges(data=True):
            if 'transaction_count' in data and 'days_active' in data:
                velocity = data['transaction_count'] / max(data['days_active'], 1)
                velocities.append(velocity)
        return np.mean(velocities) if velocities else 0
    
    def _calculate_amount_concentration(self, subgraph: nx.Graph) -> float:
        """Calculate how concentrated transaction amounts are"""
        amounts = []
        for u, v, data in subgraph.edges(data=True):
            if 'total_amount' in data:
                amounts.append(data['total_amount'])
        
        if len(amounts) < 2:
            return 0
        
        # Gini coefficient as concentration measure
        amounts = sorted(amounts)
        n = len(amounts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * amounts)) / (n * np.sum(amounts)) - (n + 1) / n
        return gini
    
    def _calculate_community_fraud_score(self, metrics: Dict) -> float:
        """
        Calculate fraud score based on community metrics
        
        High fraud indicators:
        - High density (tightly connected)
        - High transaction velocity
        - High amount concentration
        - Moderate size (not too small, not too large)
        """
        score = 0.0
        
        # Density score (higher is more suspicious)
        score += metrics['density'] * 0.3
        
        # Velocity score (higher is more suspicious)
        normalized_velocity = min(metrics['transaction_velocity'] / 10, 1.0)
        score += normalized_velocity * 0.3
        
        # Concentration score (higher is more suspicious)
        score += metrics['amount_concentration'] * 0.2
        
        # Size score (moderate sizes are more suspicious)
        size_score = 1 - abs(metrics['size'] - 5) / 10  # Peak at size 5
        size_score = max(0, size_score)
        score += size_score * 0.2
        
        return min(score, 1.0)


class GraphFraudFeatureExtractor:
    """
    Extracts graph-based features for integration with traditional ML models
    """
    
    def __init__(self):
        self.feature_cache = {}
        
    def extract_node_features(self, graph: nx.Graph, node_id: str) -> Dict:
        """
        Extract comprehensive graph-based features for a single node
        """
        if node_id not in graph.nodes():
            return self._get_default_features()
        
        features = {}
        
        # Basic centrality measures
        features.update(self._calculate_centrality_features(graph, node_id))
        
        # Neighborhood features
        features.update(self._calculate_neighborhood_features(graph, node_id))
        
        # Structural features
        features.update(self._calculate_structural_features(graph, node_id))
        
        # Community features
        features.update(self._calculate_community_features(graph, node_id))
        
        return features
    
    def _calculate_centrality_features(self, graph: nx.Graph, node_id: str) -> Dict:
        """Calculate various centrality measures"""
        try:
            degree_centrality = nx.degree_centrality(graph).get(node_id, 0)
            betweenness_centrality = nx.betweenness_centrality(graph, k=min(100, len(graph))).get(node_id, 0)
            closeness_centrality = nx.closeness_centrality(graph).get(node_id, 0)
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000).get(node_id, 0)
            
            return {
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'closeness_centrality': closeness_centrality,
                'eigenvector_centrality': eigenvector_centrality
            }
        except:
            return {
                'degree_centrality': 0,
                'betweenness_centrality': 0,
                'closeness_centrality': 0,
                'eigenvector_centrality': 0
            }
    
    def _calculate_neighborhood_features(self, graph: nx.Graph, node_id: str) -> Dict:
        """Calculate neighborhood-based features"""
        neighbors = list(graph.neighbors(node_id))
        
        return {
            'degree': len(neighbors),
            'clustering_coefficient': nx.clustering(graph, node_id),
            'neighbor_degree_mean': np.mean([graph.degree(neighbor) for neighbor in neighbors]) if neighbors else 0,
            'neighbor_degree_std': np.std([graph.degree(neighbor) for neighbor in neighbors]) if neighbors else 0,
            'triangles': nx.triangles(graph, node_id)
        }
    
    def _calculate_structural_features(self, graph: nx.Graph, node_id: str) -> Dict:
        """Calculate structural features"""
        # Core number
        core_number = nx.core_number(graph).get(node_id, 0)
        
        # K-core membership
        k_core = max([k for k in range(core_number + 1) 
                     if node_id in nx.k_core(graph, k).nodes()]) if core_number > 0 else 0
        
        return {
            'core_number': core_number,
            'k_core': k_core
        }
    
    def _calculate_community_features(self, graph: nx.Graph, node_id: str) -> Dict:
        """Calculate community-based features"""
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(graph)
            community_id = communities.get(node_id, -1)
            
            # Community size
            community_size = sum(1 for node, comm in communities.items() if comm == community_id)
            
            return {
                'community_id': community_id,
                'community_size': community_size
            }
        except:
            return {
                'community_id': -1,
                'community_size': 0
            }
    
    def _get_default_features(self) -> Dict:
        """Return default features for nodes not in graph"""
        return {
            'degree_centrality': 0,
            'betweenness_centrality': 0,
            'closeness_centrality': 0,
            'eigenvector_centrality': 0,
            'degree': 0,
            'clustering_coefficient': 0,
            'neighbor_degree_mean': 0,
            'neighbor_degree_std': 0,
            'triangles': 0,
            'core_number': 0,
            'k_core': 0,
            'community_id': -1,
            'community_size': 0
        }


class GraphNeuralNetworkManager:
    """
    Main manager class for graph neural network operations
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph_builder = TransactionGraphBuilder(config)
        self.feature_extractor = GraphFraudFeatureExtractor()
        self.community_detector = CommunityFraudDetector()
        self.models = {}
        
    def build_and_analyze_graphs(self, transactions_df: pd.DataFrame) -> Dict:
        """
        Build all types of graphs and perform comprehensive analysis
        """
        logger.info("Building and analyzing transaction graphs...")
        
        results = {}
        
        # Build different types of graphs
        customer_graph = self.graph_builder.build_customer_transaction_graph(transactions_df)
        device_graph = self.graph_builder.build_device_sharing_graph(transactions_df)
        
        results['graphs'] = {
            'customer_transaction': customer_graph,
            'device_sharing': device_graph
        }
        
        # Community detection
        communities = self.community_detector.detect_communities(customer_graph)
        community_scores = self.community_detector.score_communities_for_fraud(
            customer_graph, communities
        )
        
        results['communities'] = {
            'assignments': communities,
            'fraud_scores': community_scores
        }
        
        # Extract features for all customers
        customer_features = {}
        for customer_id in transactions_df['customer_id'].unique():
            features = self.feature_extractor.extract_node_features(customer_graph, customer_id)
            customer_features[customer_id] = features
        
        results['node_features'] = customer_features
        
        logger.info("Graph analysis completed successfully")
        return results
    
    def get_customer_graph_features(self, customer_id: str, transactions_df: pd.DataFrame) -> Dict:
        """
        Get graph-based features for a specific customer
        """
        # Build graph if not cached
        graph_key = f"customer_graph_{len(transactions_df)}"
        if graph_key not in self.feature_extractor.feature_cache:
            customer_graph = self.graph_builder.build_customer_transaction_graph(transactions_df)
            self.feature_extractor.feature_cache[graph_key] = customer_graph
        else:
            customer_graph = self.feature_extractor.feature_cache[graph_key]
        
        # Extract features
        return self.feature_extractor.extract_node_features(customer_graph, customer_id)