import pandas as pd
from pathlib import Path

AIIMS_FILE = Path("FINAL SEAT MATRIX FOR AIIMS_BHU_JIMPER ROUND 1 UG 2025 (MBBS & BDS).xlsx")

def main():
    xls = pd.ExcelFile(AIIMS_FILE)
    sheet = xls.sheet_names[0]
    raw = pd.read_excel(AIIMS_FILE, sheet_name=sheet, header=None)
    # keep first 40 rows x 30 columns to inspect
    preview = raw.iloc[:40, :30]
    preview.to_csv("aiims_preview_top.csv", index=False, header=False)
    print("Wrote aiims_preview_top.csv (first 40 rows x 30 cols).")
    print("Open it to eyeball where 'Institute' / category headers are.")

if __name__ == "__main__":
    main()

