import subprocess, sys, pathlib
H = pathlib.Path(__file__).resolve().parent
NEED = not (H / "data" / "powercase_survey_clean.csv").exists() or not (H / "models" / "intent_classifier.joblib").exists()
STEPS = ["02_generate_survey_data.py", "03_clean_transform.py", "04_analysis.py", "06_report_numbers.py"]
if NEED: _ = [subprocess.run([sys.executable, str(H / s)], check=True, cwd=str(H)) for s in STEPS]
exec(compile((H / "05_dashboard.py").read_text(), str(H / "05_dashboard.py"), "exec"))
