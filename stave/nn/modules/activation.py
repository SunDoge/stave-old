from build.lib.stave.nn.modules.module import Module
from jax.interpreters.xla import DeviceArray
import ..functional as F

Tensor = DeviceArray


class ReLU(Module):
    
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)





    
