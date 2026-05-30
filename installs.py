"""Do not install packages at app runtime.

Use requirements.txt so Streamlit Cloud installs dependencies before
the app starts.
"""

if __name__ == "__main__":
    print("Dependencies are managed by requirements.txt.")