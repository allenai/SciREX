from typing import List, Dict, Tuple
import numpy as np

def compute_coref_distance(spans: List[Tuple(int, int, str)]) -> Dict[str, Dict[Tuple(Tuple(int, int), Tuple(int, int)), float]] :
    pass

def compute_relation_distance(spans: List[Tuple(int, int, str)]) -> Dict[Tuple(Tuple(int, int), Tuple(int, int)), float]:
    pass

def compute_clusters(distance_matrix: Dict[Tuple(Tuple(int, int), Tuple(int, int)), float]) -> List[List[Tuple(int, int)]]:
    pass