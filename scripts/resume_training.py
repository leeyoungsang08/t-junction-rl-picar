from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import main
if __name__ == "__main__":
    if "--resume" not in sys.argv:
        raise SystemExit("Use --output ORIGINAL_RUN --resume ORIGINAL_RUN/checkpoints/training_checkpoint.pt")
    main()
