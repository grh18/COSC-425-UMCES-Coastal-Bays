"""
xlsx_to_csv.py
Converts each sheet in every .xlsx file to a separate .csv file.
Output files are named: <excel_filename>_<sheet_name>.csv
"""

import pandas as pd
import os
import re

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = "."          # folder containing your .xlsx files
OUTPUT_DIR = "csv_output" # folder where CSVs will be saved
# ───────────────────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """Remove characters that are unsafe in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()

def convert_all(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    xlsx_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".xlsx")]

    if not xlsx_files:
        print("No .xlsx files found in", os.path.abspath(input_dir))
        return

    total_sheets = 0

    for filename in xlsx_files:
        filepath = os.path.join(input_dir, filename)
        base     = os.path.splitext(filename)[0]

        print(f"\nProcessing: {filename}")

        try:
            xl = pd.ExcelFile(filepath)
        except Exception as e:
            print(f"  Could not open file: {e}")
            continue

        for sheet in xl.sheet_names:
            # Only export sheets whose name matches "Sheet3" / "sheet 3" variants
            if sheet.lower().replace(" ", "") != "sheet3":
                print(f"  [SKIP] {sheet}")
                continue
            try:
                df = xl.parse(sheet)
                out_name = f"{sanitize(base)}_{sanitize(sheet)}.csv"
                out_path = os.path.join(output_dir, out_name)
                df.to_csv(out_path, index=False)
                print(f"  [OK] {sheet}  ->  {out_name}")
                total_sheets += 1
            except Exception as e:
                print(f"  [FAIL] Could not convert sheet '{sheet}': {e}")

    print(f"\nDone. {total_sheets} CSV files saved to '{os.path.abspath(output_dir)}'")

if __name__ == "__main__":
    convert_all(INPUT_DIR, OUTPUT_DIR)
