"""Basic, CPU-first AutoML tuning agent for EnNeuro examples."""

from .executors import MNISTExecutor, SyntheticObjectiveExecutor, make_tiny_mnist
from .car_executor import CarBehavioralCloningExecutor, ZipSteeringDataset, read_steering_rows
from .logger import EventLogger, JSONReporter
from .protocol import AgentEvent, Executor, SearchResult, SearchSpace, TrialResult
from .search import AutoMLAgent, SequentialRandomSearch, SequentialRandomSearcher

__all__ = [
    "AgentEvent", "Executor", "EventLogger", "JSONReporter", "MNISTExecutor",
    "SearchResult", "SearchSpace", "SequentialRandomSearch", "SequentialRandomSearcher",
    "AutoMLAgent", "SyntheticObjectiveExecutor", "TrialResult", "make_tiny_mnist",
    "CarBehavioralCloningExecutor", "ZipSteeringDataset", "read_steering_rows",
]
