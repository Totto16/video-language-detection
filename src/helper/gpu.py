import contextlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cmp_to_key
from typing import Any, Optional, Self, assert_never, cast, override

import pyopencl as opencl
import torch

from content.general import MissingOverrideError
from helper.result import Result
from helper.timestamp import parse_int_safely


class GPUType(Enum):
    dedicated = "dedicated"
    integrated = "integrated"


class GPUVendor(Enum):
    nvidia = "nvidia"
    amd = "amd"
    intel = "intel"


@dataclass
class GPUDevice:
    name: str
    unique_id: str
    origin: str
    vendor: GPUVendor
    type: GPUType
    # optional properties
    memory_amount: Optional[int]  # bigger is better
    num_compute_units: Optional[int]  # bigger is better


GetDevicesResult = Result[list[GPUDevice], str]


def list_gpus_linux() -> GetDevicesResult:
    if sys.platform != "linux":
        return GetDevicesResult.err(f"Not supported on {sys.platform}")

    try:
        result = subprocess.check_output(["lspci", "-nn"]).decode()  # noqa: S607
        devices: list[GPUDevice] = []

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

                name = line[len(unique_id) + 1 :]

                if vendor is None:
                    continue

                device: GPUDevice = GPUDevice(
                    name=name,
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
    except Exception as err:  # noqa:  BLE001
        return GetDevicesResult.err(str(err))


class NvmlMemoryPy:
    total: int
    free: int
    used: int


def list_gpus_nvidia() -> GetDevicesResult:
    try:
        import pynvml  # type: ignore[import-not-found,unused-ignore]  # noqa: PLC0415

        try:
            pynvml.nvmlInit()
            devices: list[GPUDevice] = []

            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                name: str = pynvml.nvmlDeviceGetName(handle)

                num_compute_units = pynvml.nvmlDeviceGetNumGpuCores(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                memory_amount: Optional[int] = None
                if isinstance(
                    mem_info, (pynvml.c_nvmlMemory_t, pynvml.c_nvmlMemory_v2_t),
                ):
                    memory_amount = cast(NvmlMemoryPy, mem_info).total

                unique_id: str = str(pynvml.nvmlDeviceGetUUID(handle))

                device: GPUDevice = GPUDevice(
                    name=name,
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
        except Exception as err:  # noqa:  BLE001
            return GetDevicesResult.err(str(err))
        finally:
            with contextlib.suppress(Exception):
                # this can fail, but here we just ignore it
                pynvml.nvmlShutdown()
    except ImportError:
        return GetDevicesResult.err("not build with nvidia support")


@dataclass
class DeviceTopologyAmdSmi:
    domain: int
    bus: int
    device: int
    function: int

    def to_str(self: Self) -> str:
        return f"{self.domain:04x}:{self.bus:02x}:{self.device:02x}.{self.function:x}"

    @staticmethod
    def from_str(topology: str) -> Optional["DeviceTopologyAmdSmi"]:
        parsed = topology.split(":")
        if (len(parsed)) != 3:
            return None

        domain = parse_int_safely(parsed[0], 16)
        bus = parse_int_safely(parsed[1], 16)

        if domain is None or bus is None:
            return None

        parsed2 = parsed[2].split(".")

        device: Optional[int] = None
        function: Optional[int] = None

        if len(parsed2) == 1:
            device = parse_int_safely(parsed2[0], 16)
            function = 0
        elif len(parsed2) == 2:
            device = parse_int_safely(parsed2[0], 16)
            function = parse_int_safely(parsed2[1], 16)
        else:
            return None

        if device is None or function is None:
            return None

        return DeviceTopologyAmdSmi(
            domain=domain,
            bus=bus,
            device=device,
            function=function,
        )


def list_gpus_amd() -> GetDevicesResult:

    try:
        import amdsmi.amdsmi_wrapper as amdsmi  # type: ignore[import-not-found,unused-ignore]  # noqa: PLC0415
        from amdsmi import (  # type: ignore[import-not-found,unused-ignore]  # noqa: PLC0415
            amdsmi_exception,
            amdsmi_interface,
        )

        # currently not in the source code, but the correct value for this
        amdsmi_vram_type_ddr5 = amdsmi.AMDSMI_VRAM_TYPE_DDR4 + 1

        def get_gpu_type_by_ex(
            processor_handle: amdsmi_interface.processor_handle,
        ) -> GPUType:
            try:

                # these fail on integrated devices!
                _pci_bw = amdsmi_interface.amdsmi_get_gpu_pci_bandwidth(
                    processor_handle,
                )

                _bus = amdsmi_interface.amdsmi_get_pcie_info(processor_handle)
                return GPUType.dedicated  # noqa: TRY300
            except amdsmi_exception.AmdSmiException:
                return GPUType.integrated

        def get_gpu_type_by_vram_type(
            processor_handle: amdsmi_interface.processor_handle,
        ) -> GPUType:
            vram = amdsmi_interface.amdsmi_get_gpu_vram_info(processor_handle)

            vram_type: int = vram["vram_type"]

            if (
                vram_type >= amdsmi.AMDSMI_VRAM_TYPE_GDDR1
                and vram_type <= amdsmi.AMDSMI_VRAM_TYPE_GDDR7
            ):
                return GPUType.dedicated

            if (
                vram_type >= amdsmi.AMDSMI_VRAM_TYPE_DDR2
                and vram_type <= amdsmi_vram_type_ddr5
            ):
                return GPUType.integrated

            msg = f"unrecognized vram type: {vram_type}"
            raise RuntimeError(msg)

        def get_gpu_type(
            processor_handle: amdsmi_interface.processor_handle,
        ) -> GPUType:
            type1 = get_gpu_type_by_ex(processor_handle)
            type2 = get_gpu_type_by_vram_type(processor_handle)

            if type1 != type2:
                msg = "failed to detect the gpu type consistently"
                raise RuntimeError(msg)

            return type1

        def get_gpu_topology(
            processor_handle: amdsmi_interface.processor_handle,
        ) -> DeviceTopologyAmdSmi:
            # see: https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html#amdsmi-get-gpu-device-bdf
            # BDFID = ((DOMAIN & 0xffffffff) << 32) | ((BUS & 0xff) << 8) | ((DEVICE & 0x1f) <<3 ) | (FUNCTION & 0x7)  # noqa: ERA001

            # [64:32]  Domain        (32 bits)
            # [31:16]  Reserved      (16 bits, ignore)
            # [15:8]   Bus           (8 bits)
            # [7:3]    Device        (5 bits)
            # [2:0]    Function      (3 bits)

            bdfid = amdsmi_interface.amdsmi_get_gpu_bdf_id(processor_handle)

            domain = (bdfid >> 32) & 0xFFFFFFFF
            bus = (bdfid >> 8) & 0xFF
            device = (bdfid >> 3) & 0x1F
            function = bdfid & 0x7

            return DeviceTopologyAmdSmi(
                domain=domain,
                bus=bus,
                device=device,
                function=function,
            )

        try:
            amdsmi_interface.amdsmi_init()
            devices: list[GPUDevice] = []

            for processor_handle in amdsmi_interface.amdsmi_get_processor_handles():

                topology = get_gpu_topology(processor_handle)

                unique_id: str = topology.to_str()

                info = amdsmi_interface.amdsmi_get_gpu_asic_info(
                    processor_handle=processor_handle,
                )

                name: str = info["market_name"]

                num_compute_units: int = info["num_compute_units"]

                vram = amdsmi_interface.amdsmi_get_gpu_vram_info(processor_handle)

                vram_size_mb: int = vram["vram_size"]

                memory_amount = vram_size_mb * 1024 * 1024

                gpu_type = get_gpu_type(processor_handle)

                device: GPUDevice = GPUDevice(
                    name=name,
                    unique_id=unique_id,
                    origin="amdsmi",
                    vendor=GPUVendor.amd,
                    type=gpu_type,
                    memory_amount=memory_amount,
                    num_compute_units=num_compute_units,
                )
                devices.append(device)

            return GetDevicesResult.ok(devices)

        except amdsmi_exception.AmdSmiException as err:
            return GetDevicesResult.err(str(err))
        except Exception as err:  # noqa:  BLE001
            return GetDevicesResult.err(str(err))
        finally:
            with contextlib.suppress(Exception):
                # this can fail, but here we just ignore it
                amdsmi.amdsmi_shut_down()
    except ImportError:
        return GetDevicesResult.err("not build with amd support")


def list_gpus_native() -> GetDevicesResult:

    ## try best detection methods in order, if no one succeeds, return an error
    methods: list[tuple[str, Callable[[], GetDevicesResult]]] = [
        ("list_gpus_nvidia", list_gpus_nvidia),
        ("list_gpus_amd", list_gpus_amd),
    ]

    fails: list[str] = []
    devices: list[GPUDevice] = []

    for name, fun in methods:
        result = fun()

        if result.is_ok():
            devices.extend(result.get_ok())
        else:
            fails.append(f"{name} failed with error: {result.get_err()}")

    if len(devices) == 0:
        return GetDevicesResult.err(
            f"All gpu detection methods failed: {", ".join(fails)}",
        )

    return GetDevicesResult.ok(devices)


def list_gpus_torch() -> GetDevicesResult:
    if not torch.cuda.is_available():
        return GetDevicesResult.err("cuda not available")

    try:
        devices: list[GPUDevice] = []

        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)

            name: str = torch.cuda.get_device_name(i)

            vendor: Optional[GPUVendor] = None
            if "AMD" in name or "Advanced Micro Devices" in name:
                vendor = GPUVendor.amd
            elif "NVIDIA" in name:
                vendor = GPUVendor.nvidia
            elif "Intel" in name:
                vendor = GPUVendor.intel

            if vendor is None:
                continue

            unique_id: str = str(properties.uuid)

            num_compute_units: int = properties.multi_processor_count

            memory_amount: int = properties.total_memory

            gpu_type = (
                GPUType.integrated if properties.is_integrated else GPUType.dedicated
            )

            device: GPUDevice = GPUDevice(
                name=name,
                unique_id=unique_id,
                origin="torch",
                vendor=vendor,
                type=gpu_type,
                memory_amount=memory_amount,
                num_compute_units=num_compute_units,
            )
            devices.append(device)

        return GetDevicesResult.ok(devices)

    except torch.cuda.AcceleratorError as err:
        return GetDevicesResult.err(str(err))
    except Exception as err:  # noqa:  BLE001
        return GetDevicesResult.err(str(err))


def list_gpus_opencl() -> GetDevicesResult:

    def has_id(dev_id: str) -> Callable[[GPUDevice], bool]:

        def has_id_impl(device: GPUDevice) -> bool:
            return device.unique_id == dev_id

        return has_id_impl

    def opencl_get_unique_name(device: opencl.Device, vendor: GPUVendor) -> str:

        # note on device topology: 0000:03:00.0
        # according to docs it means: <domain>:<bus>:<device>.<function>

        topology: str = "<no_topology>"

        if (
            "cl_amd_device_topology" in device.extensions
            or "cl_amd_device_attribute_query" in device.extensions
        ):
            if vendor != GPUVendor.amd:
                msg = "device has amd query extensions, but not recognized as amd!"
                raise RuntimeError(
                    msg,
                )

            topology_amd = device.topology_amd

            topology = DeviceTopologyAmdSmi(
                domain=0,
                bus=topology_amd.bus,
                device=topology_amd.device,
                function=topology_amd.function,
            ).to_str()
        elif "cl_nv_device_attribute_query" in device.extensions:
            if vendor != GPUVendor.nvidia:
                msg = "device has nvidia query extensions, but not recognized as amd!"
                raise RuntimeError(
                    msg,
                )

            topology = f"{device.pci_domain_id_nv}:{device.pci_slot_id_nv}.{device.pci_bus_id_nv}"

        return f"type_{device.type}_{vendor.value}_{device.name}_{topology}"

    try:
        devices: list[GPUDevice] = []

        # on my pc, amd has two platform with the exact name and the same devices (maybe because of my amdgpu and rocm drivers??, but here we filter unique devices to occur only once)
        for platform in opencl.get_platforms():
            for device in platform.get_devices(device_type=opencl.device_type.GPU):

                name: str = device.name

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

                device_to_add: GPUDevice = GPUDevice(
                    name=name,
                    unique_id=unique_id,
                    origin="opencl",
                    vendor=vendor,
                    type=type_,
                    memory_amount=memory_amount,
                    num_compute_units=num_compute_units,
                )
                devices.append(device_to_add)

        return GetDevicesResult.ok(devices)

    except Exception as err:  # noqa:  BLE001
        return GetDevicesResult.err(str(err))


def get_devices() -> GetDevicesResult:

    ## try best detection methods in order, if no one succeeds, return an error
    methods: list[tuple[str, Callable[[], GetDevicesResult]]] = [
        ("list_gpus_native", list_gpus_native),
        ("list_gpus_torch", list_gpus_torch),
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


@dataclass
class TorchDevice:
    name: str
    uuid: str
    index: int


GpuGetResult = Result["GPU", str]


class GPU:
    __device: GPUDevice

    def __init__(self: Self, device: GPUDevice) -> None:
        self.__device = device

    @staticmethod
    def get_best(*, use_integrated: bool = False) -> GpuGetResult:
        try:
            devices_res = get_devices()

            if devices_res.is_err():
                return GpuGetResult.err(devices_res.get_err())

            devices = devices_res.get_ok()

            if not use_integrated:

                def remove_integrated_devices(device: GPUDevice) -> bool:
                    return device.type != GPUType.integrated

                devices = list(filter(remove_integrated_devices, devices))

            if len(devices) == 0:
                return GpuGetResult.err("No suitable devices found")

            cu_mult = 10**18

            def get_device_score(device: GPUDevice) -> int:
                if device.num_compute_units is None:
                    if device.memory_amount is None:
                        return 0

                    return device.memory_amount

                if device.memory_amount is None:
                    return device.num_compute_units * cu_mult

                return (device.num_compute_units * cu_mult) + device.memory_amount

            def get_best_device(device1: GPUDevice, device2: GPUDevice) -> int:

                if device1.type != device2.type:
                    return -1 if device1.type == GPUType.integrated else 1

                device1_score = get_device_score(device1)
                device2_score = get_device_score(device2)

                return device2_score - device1_score

            devices.sort(key=cmp_to_key(get_best_device))

            device = devices[0]

            if not torch.cuda.is_available():
                GpuGetResult.err("Cuda not available")

            return GpuGetResult.ok(GPU.from_device(device))

        except Exception as err:  # noqa:  BLE001
            return GpuGetResult.err(str(err))

    @staticmethod
    def from_device(device: GPUDevice) -> "GPU":
        match device.vendor:
            case GPUVendor.nvidia:
                return NvidiaGPU(device)
            case GPUVendor.amd:
                return AmdGPU(device)
            case GPUVendor.intel:
                msg = "Intel gpus not supported atm"
                raise RuntimeError(msg)
            case _:
                assert_never(device.vendor)

    def set_default_device(self: Self, device: torch.device) -> None:
        torch.set_default_device(device=device)
        torch.cuda.set_device(device=device)

    @property
    def device(self: Self) -> GPUDevice:
        return self.__device

    def empty_cache(self: Self) -> None:
        torch.cuda.empty_cache()

    def torch_device(self: Self) -> Optional[torch.device]:
        index = self.get_index()
        if index is None:
            return None
        return torch.device(type="cuda", index=index)

    def get_index(self: Self) -> Optional[int]:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            properties = torch.cuda.get_device_properties(i)
            uuid = str(properties.uuid)

            torch_device: TorchDevice = TorchDevice(
                name=name,
                uuid=uuid,
                index=i,
            )
            if self.is_eq_to_torch_device(torch_device):
                return i

        return None

    def is_eq_to_torch_device(
        self: Self,
        torch_device: TorchDevice,  # noqa: ARG002
    ) -> bool:
        raise MissingOverrideError


class NvidiaGPU(GPU):

    def __init__(self: Self, device: GPUDevice) -> None:
        if device.vendor != GPUVendor.nvidia:
            msg = "Tried to initialize nvidia gpu with wrong device"
            raise RuntimeError(msg)
        super().__init__(device)

    @override
    def is_eq_to_torch_device(self: Self, torch_device: TorchDevice) -> bool:
        msg = "Not yet implemented for nvidia"
        raise RuntimeError(msg)


class AmdGPU(GPU):

    def __init__(self: Self, device: GPUDevice) -> None:
        if device.vendor != GPUVendor.amd:
            msg = "Tried to initialize amd gpu with wrong device"
            raise RuntimeError(msg)
        super().__init__(device)

    @override
    def is_eq_to_torch_device(self: Self, torch_device: TorchDevice) -> bool:
        match self.device.origin:
            case "amdsmi":

                topo1 = DeviceTopologyAmdSmi.from_str(self.device.unique_id)

                if topo1 is None:
                    return False

                properties: Any = torch.cuda.get_device_properties(torch_device.index)

                if properties.pci_bus_id is None:
                    return False

                topo2 = DeviceTopologyAmdSmi(
                    domain=properties.pci_domain_id,
                    bus=properties.pci_bus_id,
                    device=properties.pci_device_id,
                    function=0,
                )

                return topo1 == topo2
            case "torch":
                return self.device.unique_id == torch_device.uuid
            case origin:
                msg = f"Can't compare a gpu device from origin '{origin}' with a torch device, not implemented yet"
                raise RuntimeError(msg)
