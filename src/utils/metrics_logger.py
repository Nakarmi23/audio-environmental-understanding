import csv
from pathlib import Path


class MetricsLogger:
    # Simple CSV logger for training metrics
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.initialized = False

    def log(self, metrics: dict):
        # Write metrics as a new row in CSV file
        if not self.initialized:
            # Create file and write header on first call
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writeheader()
            self.initialized = True

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)
