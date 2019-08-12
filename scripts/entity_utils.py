import os
from collections import namedtuple
from itertools import combinations

BASEPATH = os.getenv("RESULT_EXTRACTION_BASEPATH", ".")

available_entity_types_sciERC = ["Material", "Metric", "Task", "Generic", "OtherScientificTerm", "Method"]
map_available_entity_to_true = {"Material": "dataset", "Metric": "metric", "Task": "task", "Method": "model_name"}
map_true_entity_to_available = {v: k for k, v in map_available_entity_to_true.items()}

used_entities = list(map_available_entity_to_true.keys())
true_entities = list(map_available_entity_to_true.values())

Relation = namedtuple("Relation", used_entities + ["score"])

binary_relations = list(combinations(used_entities, 2))
