"""
Graph Data Generator - Synthetic graph data generation
"""

import asyncio
import time
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .base import BaseGenerator, GeneratorConfig, GenerationResult
from ..utils.logger import get_logger


@dataclass
class GraphGeneratorConfig(GeneratorConfig):
    """Configuration for graph generator."""
    num_nodes: int = 100
    avg_degree: float = 5.0
    node_features: List[str] = None
    edge_features: List[str] = None
    graph_type: str = "random"  # random, small_world, scale_free
    
    def __post_init__(self):
        super().__post_init__()
        if self.node_features is None:
            self.node_features = ["type", "value"]
        if self.edge_features is None:
            self.edge_features = ["weight"]


class GraphGenerator(BaseGenerator):
    """Simple graph data generator with various graph types."""
    
    def __init__(self, config: Optional[GraphGeneratorConfig] = None, logger=None):
        if config is None:
            config = GraphGeneratorConfig(name="graph_generator", type="graph")
        super().__init__(config, logger)
    
    async def initialize(self) -> None:
        if self.initialized:
            return
        self.logger.info("Initializing GraphGenerator...")
        self.initialized = True
        self.logger.info("GraphGenerator initialized")
    
    async def generate(self, num_records: int, seed: Optional[int] = None, **kwargs) -> GenerationResult:
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        start_time = time.time()
        graphs = []
        
        for _ in range(num_records):
            graph = await self._generate_single_graph()
            graphs.append(graph)
        
        generation_time = time.time() - start_time
        await self._update_statistics(generation_time, num_records)
        
        return GenerationResult(
            data=graphs,
            metadata={
                'generation_time': generation_time,
                'num_records': num_records,
                'num_nodes': self.config.num_nodes,
                'avg_degree': self.config.avg_degree
            }
        )
    
    async def _generate_single_graph(self) -> Dict:
        """Generate a single synthetic graph."""
        num_nodes = self.config.num_nodes
        
        # Generate nodes
        nodes = []
        for i in range(num_nodes):
            node = {'id': i}
            for feature in self.config.node_features:
                if feature == 'type':
                    node[feature] = random.choice(['A', 'B', 'C'])
                elif feature == 'value':
                    node[feature] = random.uniform(0, 1)
                else:
                    node[feature] = random.random()
            nodes.append(node)
        
        # Generate edges
        edges = []
        target_edges = int(num_nodes * self.config.avg_degree / 2)
        
        for _ in range(target_edges):
            source = random.randint(0, num_nodes - 1)
            target = random.randint(0, num_nodes - 1)
            
            if source != target:
                edge = {'source': source, 'target': target}
                for feature in self.config.edge_features:
                    if feature == 'weight':
                        edge[feature] = random.uniform(0, 1)
                    else:
                        edge[feature] = random.random()
                edges.append(edge)
        
        return {'nodes': nodes, 'edges': edges}
    
    def validate_config(self) -> List[str]:
        issues = []
        if self.config.num_nodes <= 0:
            issues.append("num_nodes must be positive")
        if self.config.avg_degree < 0:
            issues.append("avg_degree cannot be negative")
        return issues
    
    async def cleanup(self) -> None:
        self.initialized = False