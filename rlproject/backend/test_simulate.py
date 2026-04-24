"""Quick test to verify simulate mode works."""
from eval_engine import EvalEngine

e = EvalEngine(simulate=True)
print("Simulate flag:", e.simulate)
result = e.connect()
print("Connect result:", result)
print("Expected: True")

# Check if we can start eval without hardware errors
import os
print("RL model path exists:", os.path.exists(e.rl_model_path))
print("YOLO model path exists:", os.path.exists(e.yolo_model_path))
