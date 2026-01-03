from enum import Enum
from logging import Logger
from typing import Optional, Self, TypeIs

import torch

from helper.gpu import GPU
from helper.log import get_logger

logger: Logger = get_logger()


class AllocatorType(Enum):
    cpu = "cpu"
    gpu = " gpu"


class Allocator:
    type: AllocatorType

    def __init__(self: Self, type_: AllocatorType) -> None:
        self.type = type_


class CPUAllocator(Allocator):
    def __init__(self) -> None:
        super().__init__(type_=AllocatorType.cpu)


class GPUAllocator(Allocator):
    gpu: GPU

    def __init__(self, gpu: GPU) -> None:
        super().__init__(type_=AllocatorType.gpu)
        self.gpu = gpu


def is_cpu_allocator(allocator: GPUAllocator | CPUAllocator) -> TypeIs[CPUAllocator]:
    return allocator.type == AllocatorType.cpu


class DeviceManager:
    __device_allocator: CPUAllocator | GPUAllocator

    def __init__(self: Self) -> None:
        self.__init_type()

    def force_cpu(self: Self) -> None:
        self.__device_allocator = CPUAllocator()

    def __init_type(self: Self) -> None:
        self.__device_allocator = CPUAllocator()

        gpu_result = GPU.get_best(use_integrated=False)

        if gpu_result.is_err():
            logger.warning("Got GPU error: %s", gpu_result.get_err())
            self.__device_allocator = CPUAllocator()
        else:
            self.__device_allocator = GPUAllocator(gpu_result.get_ok())

    def clear_device_cache(self: Self) -> None:
        if not is_cpu_allocator(self.__device_allocator):
            self.__device_allocator.gpu.empty_cache()

    def get_torch_device(self: Self) -> Optional[torch.device]:
        if is_cpu_allocator(self.__device_allocator):
            return torch.device(device="cpu")

        gpu = self.__device_allocator.gpu

        device = gpu.torch_device()

        if device is None:
            return None

        """ not possible, as speechbrain uses some hardcoded cpu device wrongly in the code:
        ```python
        zero = torch.zeros(1, device=self.device_inp)
        fbank_matrix = torch.max(
            zero, torch.min(left_side, right_side)
        ).transpose(0, 1)
        ```
        this uses cpu and default allocated tensors, so this fails :(
        note, device_inp is hardcoded to cpu, and caN#t be set
        see: speechbrain/processing/features.py:642
        """

        # gpu.set_default_device(device)  # noqa: ERA001

        # this can be used, as it specifies the default cuda device
        gpu.set_default_cuda_device(device=device)

        return device

    @property
    def type(self: Self) -> AllocatorType:
        return self.__device_allocator.type
