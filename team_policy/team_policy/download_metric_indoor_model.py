from pathlib import Path

from huggingface_hub import snapshot_download


repo_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
root = Path(__file__).resolve().parent
local_dir = root / "models" / "Depth-Anything-V2-Metric-Indoor-Base-hf"
local_dir.parent.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
print(local_dir)
