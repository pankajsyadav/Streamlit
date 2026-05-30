import subprocess
import sys

subprocess.check_call(
	[sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "streamlit", "pandas", "numpy", "ipykernel"]
)
subprocess.check_call(
	[sys.executable, "-m", "pip", "install", "ipykernel", "-U", "--user", "--force-reinstall"]
)