import csv
from pathlib import Path


class MetricsLogger:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.initialized = False

    def log(self, metrics: dict):
        """
        metrics: dict of scalar values
        """
        if not self.initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writeheader()
            self.initialized = True

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)
