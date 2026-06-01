import numpy as np
from dataclasses import dataclass, field


@dataclass
class OptResult:
    x:          np.ndarray
    f_val:      float
    n_iter:     int
    n_f:        int
    n_grad:     int
    n_hess:     int
    converged:  bool
    status:     str
    trajectory: list = field(default_factory=list)

    def __str__(self):
        mark = "✓" if self.converged else "✗"
        return (f"{mark} iter={self.n_iter:5d}  f={self.n_f:6d}  "
                f"grad={self.n_grad:5d}  hess={self.n_hess:5d}  "
                f"f(x*)={self.f_val:.4e}  x*={np.round(self.x, 6)}")
