from dataclasses import dataclass
from enum import Enum
from typing import Optional, Self, cast
import pynvml
from torch import cuda
from helper.result import Result
import subprocess
import sys
from collections.abc import Callable


class GPUType(Enum):
    dedicated = "dedicated"
    integrated = "integrated"


class GPUVendor(Enum):
    nvidia = "nvidia"
    amd = "amd"
    intel = "intel"


@dataclass
class Device:
    vendor: Optional[GPUVendor]
    type: GPUType
    memory_amount: Optional[int]  # bigger is better
    num_compute_units: Optional[int]  # bigger is better


GetDevicesResult = Result[list[Device], str]


def list_gpus_linux() -> GetDevicesResult:
    if sys.platform != "linux":
        return GetDevicesResult.err(f"Not supported on {sys.platform}")

    try:
        result = subprocess.check_output(["lspci", "-nn"]).decode()
        devices: list[Device] = []

        for line in result.splitlines():

            if "VGA compatible controller" in line or "3D controller" in line:
                vendor: Optional[GPUVendor] = None
                if "NVIDIA" in line:
                    vendor = GPUVendor.nvidia
                elif "AMD" in line or "ATI" in line:
                    vendor = GPUVendor.amd
                elif "Intel" in line:
                    vendor = GPUVendor.intel

                type_ = (
                    GPUType.integrated
                    if (vendor == GPUVendor.intel or "APU" in line)
                    else GPUType.dedicated
                )

                device: Device = Device(
                    vendor=vendor,
                    type=type_,
                    memory_amount=None,
                    num_compute_units=None,
                )
                devices.append(device)

        return GetDevicesResult.ok(devices)

    except subprocess.CalledProcessError as err:
        return GetDevicesResult.err(str(err))
    except subprocess.SubprocessError as err:
        return GetDevicesResult.err(str(err))
    except Exception as err:
        return GetDevicesResult.err(str(err))


class nvmlMemoryPy:
    total: int
    free: int
    used: int


def list_gpus_nvidia() -> GetDevicesResult:
    try:
        pynvml.nvmlInit()
        devices: list[Device] = []

        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            num_compute_units = pynvml.nvmlDeviceGetNumGpuCores(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            memory_amount: Optional[int] = None
            if isinstance(mem_info, pynvml.c_nvmlMemory_t) or isinstance(
                mem_info, pynvml.c_nvmlMemory_v2_t
            ):
                memory_amount = cast(nvmlMemoryPy, mem_info).total

            device: Device = Device(
                vendor=GPUVendor.nvidia,
                type=GPUType.dedicated,
                memory_amount=memory_amount,
                num_compute_units=num_compute_units,
            )
            devices.append(device)

        pynvml.nvmlShutdown()
        return GetDevicesResult.ok(devices)

    except pynvml.NVMLError as err:
        return GetDevicesResult.err(str(err))
    except Exception as err:
        return GetDevicesResult.err(str(err))


def list_gpus_amd() -> GetDevicesResult:
    pass


def list_gpus_native() -> GetDevicesResult:
    pass


def get_devices() -> GetDevicesResult:

    ## try best detection methods in order, if no one succeeds, return an error
    methods: list[tuple[str, Callable[[], GetDevicesResult]]] = [
        ("list_gpus_native", list_gpus_native),
        ("list_gpus_opencl", list_gpus_opencl),
        ("list_gpus_linux", list_gpus_linux),
    ]

    fails: list[str] = []

    for name, fun in methods:
        result = fun()

        if result.is_ok():
            return GetDevicesResult.ok(result.get_ok())

        fails.append(f"{name} failed with error: {result.get_err()}")

    return GetDevicesResult.err(f"All gpu detection methods failed: {", ".join(fails)}")


GpuGetResult = Result["GPU", str]


class GPU:
    type: GPUType
    vendor: GPUVendor

    def __init__(self: Self, type_: GPUType, vendor: GPUVendor) -> None:
        self.type = type_
        self.vendor = vendor

    @staticmethod
    def get_best(*, use_integrated: bool = False) -> GpuGetResult:

        devices = get_devices()


class NvidiaGPU(GPU):

    @staticmethod
    def print_gpu_stat() -> None:
        if not cuda.is_available():
            return

        ClassifierManager.clear_gpu_cache()

        nvmlInit()
        # TODO: support more than one
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        logger.info("GPU stats:")
        logger.info("total : %s", naturalsize(info.total, binary=True))
        logger.info("free : %s", naturalsize(info.free, binary=True))
        logger.info("used : %s", naturalsize(info.used, binary=True))

    def empty_cache():
        cuda.empty_cache()

    def device_name_for_torch():
        return f"cuda:{index}"


# cuda.is_available()

GPUErrors: list[RuntimeError] = [
    cuda.OutOfMemoryError,
    ## TODO: add amd OOm error
]
