"""External baselines for dialogue topic segmentation."""

from hi_ontop.baselines.graphseg_window import GraphSegWindowD
from hi_ontop.baselines.greedyseg_delay2 import GreedySegOnlineDelay2
from hi_ontop.baselines.texttiling_streaming import StreamingTextTiling

__all__ = ["StreamingTextTiling", "GreedySegOnlineDelay2", "GraphSegWindowD"]
