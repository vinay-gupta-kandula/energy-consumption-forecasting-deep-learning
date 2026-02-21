import subprocess
import os

def run(step):
    print(f"\n========== RUNNING: {step} ==========")
    result = subprocess.run(step, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step}")

def main():

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. preprocessing
    run("python -m src.preprocess")

    # 2. feature engineering
    run("python -m src.feature_engineering")

    # 3. training
    run("python -m src.train")

    # 4. evaluation
    run("python -m src.evaluate")

    print("\nPIPELINE FINISHED SUCCESSFULLY")

if __name__ == "__main__":
    main()
