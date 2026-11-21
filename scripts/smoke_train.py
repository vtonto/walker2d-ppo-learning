from pathlib import Path

from walker2d_ppo.config import profile
from walker2d_ppo.trainer import train


if __name__ == "__main__":
    checkpoint = train(profile("smoke", output_dir=Path("runs/smoke")))
    print(f"smoke test passed: {checkpoint}")

