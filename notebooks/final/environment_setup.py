# Kaggle's current image ships torch 2.10+cu128, whose wheels dropped Pascal (sm_60) kernels.
# P100 = sm_60 -> "no kernel image is available". T4 = sm_75 -> fine as-is.
# This cell detects the GPU BEFORE importing torch and, on Pascal, downgrades to
# torch 2.6.0+cu124 (last widely-used Kaggle pairing that ships sm_60 kernels),
# then builds detectron2 against whichever torch is active.
import subprocess

def sh(cmd, note=""):
    if note:
        print(note)
    print(">>", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    print("   " + "; ".join(tail[-4:]) if tail else "   (no output)")
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-3000:])
        raise RuntimeError(f"command failed: {cmd}")

q = subprocess.run("nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader",
                   shell=True, capture_output=True, text=True).stdout.strip().splitlines()
cap_s, gpu_name = ("7.5", "unknown")
if q:
    parts = [p.strip() for p in q[0].split(",")]
    cap_s = parts[0]
    gpu_name = parts[1] if len(parts) > 1 else "unknown"
try:
    CAP = float(cap_s)
except ValueError:
    CAP = 6.0 if "P100" in gpu_name.upper() else 7.5
print(f"GPU: {gpu_name} | compute capability {CAP}")

sh("pip install -q 'setuptools<81'")
if CAP < 7.0:
    sh("pip install -q torch==2.6.0 torchvision==0.21.0"
       " --index-url https://download.pytorch.org/whl/cu124",
       note="Pascal GPU: downgrading torch to a build with sm_60 kernels (~2.5 GB, 3-6 min)")
sh("pip install -q --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'",
   note="building detectron2 against the active torch (~5-8 min)")

import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| arch list:", torch.cuda.get_arch_list())
try:
    val = float((torch.ones(8, device="cuda") * 2).sum().item())  # real kernel launch
    print(f"CUDA smoke test OK ({val}) on", torch.cuda.get_device_name(0))
except Exception as e:
    raise SystemExit(f"CUDA smoke test FAILED on {torch.cuda.get_device_name(0)}: {e}."
                     " Switch Settings -> Accelerator -> GPU T4 x2 and rerun.")
