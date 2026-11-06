from __future__ import annotations

from typing import Any, Protocol

import torch


class CausalLMProtocol(Protocol):
    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor: ...

    def to(
        self,
        device: str | torch.device,
    ) -> CausalLMProtocol: ...

    def eval(self) -> CausalLMProtocol: ...
