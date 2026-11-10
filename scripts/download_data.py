"""Download the OpenML Electricity benchmark to data/electricity.csv."""

from epoch_research.data import download_electricity

if __name__ == "__main__":
    path = download_electricity()
    print(f"Dataset saved: {path}")
