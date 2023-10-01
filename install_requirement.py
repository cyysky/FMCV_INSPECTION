import subprocess
import sys

requirements_file = "requirement.txt"

with open(requirements_file, "r") as f:
    requirements = f.readlines()

for requirement in requirements:
    requirement = requirement.strip()

    try:
        print(f"Installing {requirement}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
    except subprocess.CalledProcessError:
        print(f"Failed to install {requirement}, skipping...")
        continue