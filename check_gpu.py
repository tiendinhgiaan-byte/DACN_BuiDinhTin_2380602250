"""
Kiem tra moi truong PyTorch + GPU cho do an CNN-SA-RFR.
Chay: python check_gpu.py
"""

import sys
import platform


def human(n_bytes):
    return f"{n_bytes / (1024 ** 3):.2f} GB"


def main():
    print("=" * 60)
    print("KIEM TRA MOI TRUONG - DO AN CNN-SA-RFR")
    print("=" * 60)

    print(f"\n[He thong]")
    print(f"  OS          : {platform.system()} {platform.release()}")
    print(f"  Python      : {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        print("\n  LOI: chua cai PyTorch. Xem lai buoc 3.")
        return 1

    print(f"\n[PyTorch]")
    print(f"  torch       : {torch.__version__}")
    print(f"  CUDA build  : {torch.version.cuda}")

    if torch.version.cuda is None:
        print("\n  CANH BAO: ban PyTorch nay la ban CPU-only.")
        print("  Go ra va cai lai bang lenh co --index-url .../cu124")
        return 1

    if not torch.cuda.is_available():
        print("\n  LOI: PyTorch khong thay GPU.")
        print("  Kiem tra: driver NVIDIA da cap nhat chua (chay 'nvidia-smi').")
        return 1

    print(f"  cuDNN       : {torch.backends.cudnn.version()}")

    print(f"\n[GPU]")
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    print(f"  Ten         : {props.name}")
    print(f"  Compute cap : {props.major}.{props.minor}")
    print(f"  VRAM tong   : {human(props.total_memory)}")
    print(f"  SM count    : {props.multi_processor_count}")

    # Thu mot phep nhan ma tran tren GPU
    print(f"\n[Thu nghiem]")
    try:
        import time

        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 10
        tflops = 2 * (4096 ** 3) / dt / 1e12
        print(f"  Matmul 4096 : {dt * 1000:.1f} ms/lan  (~{tflops:.1f} TFLOPS)")
        del a, b
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  LOI matmul  : {e}")
        return 1

    # Thu mot lop conv giong luc train that
    try:
        import torch.nn as nn

        net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 4),
        ).cuda()
        x = torch.randn(32, 3, 224, 224, device="cuda")
        y = net(x)
        loss = y.sum()
        loss.backward()
        peak = torch.cuda.max_memory_allocated()
        print(f"  Conv fwd+bwd: OK, output {tuple(y.shape)}")
        print(f"  VRAM dinh   : {human(peak)} (batch 32 @ 224x224)")
    except Exception as e:
        print(f"  LOI conv    : {e}")
        return 1

    # Cac thu vien con lai
    print(f"\n[Thu vien khac]")
    for name in [
        "torchvision", "numpy", "pandas", "matplotlib",
        "seaborn", "sklearn", "PIL", "cv2", "albumentations", "tqdm",
    ]:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {name:<15}: {ver}")
        except ImportError:
            print(f"  {name:<15}: CHUA CAI")

    print("\n" + "=" * 60)
    print("MOI TRUONG SAN SANG.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
