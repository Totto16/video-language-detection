import subprocess


def list_gpus_linux():
    result = subprocess.check_output(["lspci", "-nn"]).decode()
    gpus = []

    for line in result.splitlines():
        if "VGA compatible controller" in line or "3D controller" in line:
            vendor = "Unknown"
            if "NVIDIA" in line:
                vendor = "NVIDIA"
            elif "AMD" in line or "ATI" in line:
                vendor = "AMD"
            elif "Intel" in line:
                vendor = "Intel"

            integrated = vendor == "Intel" or "APU" in line

            gpus.append(
                {"description": line, "vendor": vendor, "integrated": integrated}
            )

    return gpus


def list_amd_gpus():
    out = subprocess.check_output(["rocm-smi", "-i"]).decode()
    return {"vendor": "AMD", "integrated": False, "raw": out}


##print(list_amd_gpus())

import pyopencl as cl


def list_gpus_opencl():
    gpus = []

    for platform in cl.get_platforms():
        for device in platform.get_devices(device_type=cl.device_type.GPU):

            vendor = device.vendor.strip()
            name = device.name.strip()

            # Memory info
            vram_mb = device.global_mem_size // (1024 * 1024)
            
            # CL_DEVICE_MAX_COMPUTE_UNITS
            cu = device.max_compute_units

            # This is the most reliable flag for iGPU vs dGPU
            unified = device.host_unified_memory

            if unified:
                gpu_type = "Integrated"
            else:
                gpu_type = "Dedicated"

            # Normalize vendor
            if "AMD" in vendor or "Advanced Micro Devices" in vendor:
                vendor_name = "AMD"
            elif "NVIDIA" in vendor:
                vendor_name = "NVIDIA"
            elif "Intel" in vendor:
                vendor_name = "Intel"
            else:
                vendor_name = vendor

            gpus.append(
                {
                    "name": name,
                    "vendor": vendor_name,
                    "type": gpu_type,
                    "vram_mb": vram_mb,
                    "host_unified_memory": unified,
                    "platform": platform.name.strip(),
                }
            )

    return gpus


## print(list_gpus_opencl())




from amdsmi import *

try:
    amdsmi_init()

    num_of_GPUs = len(amdsmi_get_processor_handles())
    if num_of_GPUs == 0:
        print("No GPUs on machine")

    devices = amdsmi_get_processor_handles()
    if len(devices) == 0:
        print("No GPUs on machine")
    else:
        for processor_handle in devices:
            print(amdsmi_get_gpu_device_uuid(processor_handle))
            type_of_GPU = amdsmi_get_processor_type(processor_handle)
            print(type_of_GPU)
            info = amdsmi_get_gpu_asic_info(processor_handle=processor_handle)

            name = info["market_name"]

            print(name, info)
            print(info["num_compute_units"])

except AmdSmiException as e:
    print(e)
finally:
    try:
        amdsmi_shut_down()
    except AmdSmiException as e:
        print(e)
