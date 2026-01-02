from dataclasses import dataclass
from enum import Enum
from typing import Optional, Self, cast
import pynvml
from torch import cuda
from helper.result import Result
import subprocess
import sys
from collections.abc import Callable
import amdsmi.amdsmi_wrapper as amdsmi
import amdsmi.amdsmi_interface as amdsmi_interface
import amdsmi.amdsmi_exception as amdsmi_exception
import contextlib
import pyopencl as opencl


class GPUType(Enum):
    dedicated = "dedicated"
    integrated = "integrated"


class GPUVendor(Enum):
    nvidia = "nvidia"
    amd = "amd"
    intel = "intel"


@dataclass
class Device:
    unique_id: str
    origin: str
    vendor: GPUVendor
    type: GPUType
    # optional properties
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

                type_ = GPUType.integrated if ("APU" in line) else GPUType.dedicated

                unique_id = line.split(" ")[0]

                if vendor is None:
                    continue

                device: Device = Device(
                    unique_id=unique_id,
                    origin="lspci",
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

            unique_id: str = str(pynvml.nvmlDeviceGetUUID(handle))

            device: Device = Device(
                unique_id=unique_id,
                origin="nvml",
                vendor=GPUVendor.nvidia,
                type=GPUType.dedicated,
                memory_amount=memory_amount,
                num_compute_units=num_compute_units,
            )
            devices.append(device)

        return GetDevicesResult.ok(devices)

    except pynvml.NVMLError as err:
        return GetDevicesResult.err(str(err))
    except Exception as err:
        return GetDevicesResult.err(str(err))
    finally:
        with contextlib.suppress(Exception):
            # this can fail, but here we just ignore it
            pynvml.nvmlShutdown()


def list_gpus_amd() -> GetDevicesResult:

    def get_gpu_type(
        processor_handle: amdsmi_interface.processor_handle,
    ) -> GPUType:
        try:

            _id = amdsmi.amdsmi_get_gpu_bdf_id(processor_handle)
            _pci_bw = amdsmi_interface.amdsmi_get_gpu_pci_bandwidth(processor_handle)

            _bus = amdsmi_interface.amdsmi_get_pcie_info(processor_handle)

            vram = amdsmi_interface.amdsmi_get_gpu_vram_info(processor_handle)

            vram_type: int = vram["vram_type"]

            ## TODO AMDSMI_VRAM_TYPE_GDDR6
            ## TODO amdsmi_vram_type_t__enumvalues

            return GPUType.dedicated
        except amdsmi_exception.AmdSmiException:
            return GPUType.integrated

    try:
        amdsmi_interface.amdsmi_init()
        devices: list[Device] = []

        for processor_handle in amdsmi_interface.amdsmi_get_processor_handles():

            unique_id: str = amdsmi_interface.amdsmi_get_gpu_device_uuid(
                processor_handle
            )

            info = amdsmi_interface.amdsmi_get_gpu_asic_info(
                processor_handle=processor_handle
            )

            num_compute_units: int = info["num_compute_units"]

            vram = amdsmi_interface.amdsmi_get_gpu_vram_info(processor_handle)

            vram_size_mb: int = vram["vram_size"]

            memory_amount = vram_size_mb * 1024 * 1024

            gpu_type = get_gpu_type(processor_handle)

            device: Device = Device(
                unique_id=unique_id,
                origin="amdsmi",
                vendor=GPUVendor.nvidia,
                type=gpu_type,
                memory_amount=memory_amount,
                num_compute_units=num_compute_units,
            )
            devices.append(device)

        return GetDevicesResult.ok(devices)

    except amdsmi_exception.AmdSmiException as err:
        return GetDevicesResult.err(str(err))
    except Exception as err:
        return GetDevicesResult.err(str(err))
    finally:
        with contextlib.suppress(Exception):
            # this can fail, but here we just ignore it
            amdsmi.amdsmi_shut_down()


def list_gpus_native() -> GetDevicesResult:

    ## try best detection methods in order, if no one succeeds, return an error
    methods: list[tuple[str, Callable[[], GetDevicesResult]]] = [
        ("list_gpus_nvidia", list_gpus_nvidia),
        ("list_gpus_amd", list_gpus_amd),
    ]

    fails: list[str] = []
    devices: list[Device] = []

    for name, fun in methods:
        result = fun()

        if result.is_ok():
            devices.extend(result.get_ok())
        else:
            fails.append(f"{name} failed with error: {result.get_err()}")

    if len(devices) == 0:
        return GetDevicesResult.err(
            f"All gpu detection methods failed: {", ".join(fails)}"
        )

    return GetDevicesResult.ok(devices)


def list_gpus_opencl() -> GetDevicesResult:

    def has_id(id: str) -> Callable[[Device], bool]:

        def has_id_impl(device: Device) -> bool:
            return device.unique_id == id

        return has_id_impl

    def opencl_get_unique_name(device: opencl.Device, vendor: GPUVendor) -> str:

        topology: str = "<no_topology>"

        if (
            "cl_amd_device_topology" in device.extensions
            or "cl_amd_device_attribute_query" in device.extensions
        ):
            if vendor != GPUVendor.amd:
                raise RuntimeError(
                    "device has amd query extensions, but not recognized as amd!"
                )

            topology = str(device.topology_amd)
        elif "cl_nv_device_attribute_query" in device.extensions:
            if vendor != GPUVendor.nvidia:
                raise RuntimeError(
                    "device has nvidia query extensions, but not recognized as amd!"
                )

            topology = f"{device.pci_domain_id_nv}_{device.pci_slot_id_nv}_{device.pci_bus_id_nv}"

        unique_id = f"{device.type}_{str(vendor)}_{device.name}_{topology}"

        return unique_id

    try:
        devices: list[Device] = []

        # on my pc, amd has two platform with the exact name and the same devices (maybe because of my amdgpu and rocm drivers??, but here we filter unique devices to occur only once)
        for platform in opencl.get_platforms():
            for device in platform.get_devices(device_type=opencl.device_type.GPU):

                vendor_name = device.vendor.strip()

                vendor: Optional[GPUVendor] = None
                if "AMD" in vendor_name or "Advanced Micro Devices" in vendor_name:
                    vendor = GPUVendor.amd
                elif "NVIDIA" in vendor_name:
                    vendor = GPUVendor.nvidia
                elif "Intel" in vendor_name:
                    vendor = GPUVendor.intel

                if vendor is None:
                    continue

                unique_id: str = opencl_get_unique_name(device, vendor)

                if len(list(filter(has_id(unique_id), devices))) > 0:
                    continue

                # Memory info
                memory_amount = device.global_mem_size

                # CL_DEVICE_MAX_COMPUTE_UNITS
                num_compute_units = device.max_compute_units

                # This is the most reliable flag for iGPU vs dGPU
                unified = device.host_unified_memory

                type_: GPUType = GPUType.integrated if unified else GPUType.dedicated

                device_to_add: Device = Device(
                    unique_id=unique_id,
                    origin="opencl",
                    vendor=vendor,
                    type=type_,
                    memory_amount=memory_amount,
                    num_compute_units=num_compute_units,
                )
                devices.append(device_to_add)

        return GetDevicesResult.ok(devices)

    except Exception as err:
        return GetDevicesResult.err(str(err))


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
    amdsmi_exception.AmdSmiException,
]
