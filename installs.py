import subprocess
import sys

subprocess.check_call(
	[sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "streamlit", "pandas", "numpy"]
)