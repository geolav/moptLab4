from dataclasses import dataclass, field
import numpy as np
from typing import List

@dataclass
class OptimizationResult:
    """
    Единый унифицированный формат вывода результатов для всех оптимизаторов.
    """
    x_opt: np.ndarray
    f_opt: float
    iters: int
    f_evals: int
    g_evals: int
    h_evals: int
    status: str
    path: List[np.ndarray] = field(default_factory=list)