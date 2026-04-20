# gpu_check.py - исправленная версия с безопасным fallback
import os
import torch

def get_device() -> tuple[torch.device, bool]:
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    try:
        import psutil
        import humanize
        process = psutil.Process(os.getpid())
        ram_free = humanize.naturalsize(psutil.virtual_memory().available)
        proc_size = humanize.naturalsize(process.memory_info().rss)
        print(f"Gen RAM Free: {ram_free} | Proc size: {proc_size}")
    except ImportError:
        print("(psutil/humanize not installed, skipping RAM info)")
    except Exception as e:
        print(f"(Failed to get RAM info: {e})")


    try:
        import GPUtil as GPU
        gpus = GPU.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"GPU RAM Free: {gpu.memoryFree:.0f}MB | Used: {gpu.memoryUsed:.0f}MB | "
                  f"Util {gpu.memoryUtil*100:3.0f}% | Total {gpu.memoryTotal:.0f}MB")
        else:
            print("No GPUs found by GPUtil")
    except ImportError:
        print("GPUtil not installed, skipping GPU stats")
    except Exception as e:
        print(f"GPUtil error: {e}")

    print("Using device:", device)
    return device, cuda_available