"""Runs the whole pipeline in order. Usage:  python run_all.py   then   streamlit run 05_dashboard.py"""
import subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for s in ["02_generate_survey_data.py", "03_clean_transform.py", "04_analysis.py", "06_report_numbers.py"]:
    print(f"\n===== {s} =====")
    subprocess.run([sys.executable, str(HERE / s)], check=True, cwd=HERE)
print("\nDone. Now run:  streamlit run 05_dashboard.py")
