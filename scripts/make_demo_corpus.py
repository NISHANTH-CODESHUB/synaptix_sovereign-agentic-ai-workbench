"""Build a small, realistic, SYNTHETIC plant corpus for the demo.

Nothing here is a real document from any organisation. The UT report is rendered
to a PDF and then re-rasterised with rotation and noise so that the OCR path
faces genuinely degraded input rather than a clean text layer.
"""
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo_corpus"
OUT.mkdir(exist_ok=True)

W, H = 595, 842  # A4 points


def make_pdf(name: str, lines: list[tuple[str, int, bool]]) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    y = 60
    for text, size, bold in lines:
        if text == "@@PAGEBREAK":
            page = doc.new_page(width=W, height=H)
            y = 60
            continue
        font = "hebo" if bold else "helv"
        for chunk in _wrap(text, 95 if size <= 9 else 72):
            page.insert_text((56, y), chunk, fontsize=size, fontname=font)
            y += size + 4.5
        y += 3
    p = OUT / name
    doc.save(p)
    doc.close()
    return p


def _wrap(t: str, n: int) -> list[str]:
    if not t.strip():
        return [""]
    words, out, cur = t.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= n:
            cur = f"{cur} {w}".strip()
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def scan_pdf(pdf: Path, out_pdf: Path, dpi=180, angle=0.6, noise=10):
    """Rasterise EVERY page and rebuild as an image-only PDF - i.e. what a
    departmental scanner actually produces: no text layer anywhere, so OCR is
    the only way in."""
    src = fitz.open(pdf)
    dst = fitz.open()
    for pg in src:
        pix = pg.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255),
                         resample=Image.BICUBIC)
        img = img.filter(ImageFilter.GaussianBlur(0.4))
        a = np.asarray(img).astype(np.int16)
        a = a + np.random.normal(0, noise, a.shape).astype(np.int16)
        a = np.clip(a * 0.97 + 6, 0, 255).astype(np.uint8)
        tmp = out_pdf.parent / "._tmp_scan.png"
        Image.fromarray(a).convert("L").save(tmp)
        page = dst.new_page(width=W, height=H)
        page.insert_image(fitz.Rect(0, 0, W, H), filename=str(tmp))
        tmp.unlink(missing_ok=True)
    src.close()
    dst.save(out_pdf)
    dst.close()
    return out_pdf


def scan_effect(pdf: Path, out_png: Path, dpi=200, angle=0.7, noise=11):
    """Render → rotate → blur → noise → grey. Makes OCR do real work."""
    doc = fitz.open(pdf)
    pix = doc[0].get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255),
                     resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.45))
    a = np.asarray(img).astype(np.int16)
    a = a + np.random.normal(0, noise, a.shape).astype(np.int16)
    a = np.clip(a * 0.97 + 6, 0, 255).astype(np.uint8)
    Image.fromarray(a).convert("L").convert("RGB").save(out_png, quality=88)
    return out_png


# --------------------------------------------------------------------- corpus
UT_REPORT = [
    ("BHARAT REFINERY LIMITED  (SYNTHETIC DEMO DOCUMENT)", 12, True),
    ("INSPECTION DEPARTMENT - MECHANICAL", 9, False),
    ("", 8, False),
    ("ULTRASONIC THICKNESS SURVEY REPORT", 12, True),
    ("", 8, False),
    ("Report No: UT/U2/2026/0417            Date: 04.09.2026", 9, True),
    ("Unit: Unit-2 Crude Distillation       Line No: 12in-P-0421", 9, False),
    ("Service: Hot Crude Transfer           Material: A106 Gr.B", 9, False),
    ("Design Pressure: 18.5 kg/cm2          Design Temp: 340 deg C", 9, False),
    ("Nominal Thickness: 9.5 mm             Minimum Required (t-min): 6.4 mm", 9, False),
    ("Corrosion Allowance: 3.0 mm           Corrosion Rate: 0.21 mm/year", 9, False),
    ("Inspected By: R. Krishnan, Level II UT     Agency: In-house", 9, False),
    ("", 8, False),
    ("1. SCOPE", 10, True),
    ("Ultrasonic thickness measurement was carried out on 18 condition monitoring "
     "locations (CML) of line 12in-P-0421 as part of the risk based inspection "
     "programme. Measurements were taken using a calibrated digital thickness "
     "gauge at four quadrants per location and the minimum value is reported.", 9, False),
    ("", 8, False),
    ("2. THICKNESS READINGS", 10, True),
    ("CML No     Location                  Nominal    Measured   Remarks", 9, True),
    ("CML-01     Upstream elbow             9.5        8.9        Satisfactory", 9, False),
    ("CML-02     Straight run A             9.5        8.7        Satisfactory", 9, False),
    ("CML-03     Downstream of tee          9.5        8.2        Satisfactory", 9, False),
    ("CML-04     Reducer inlet              9.5        7.9        Monitor", 9, False),
    ("CML-05     Straight run B             9.5        8.4        Satisfactory", 9, False),
    ("CML-06     Elbow at rack crossing     9.5        7.1        Monitor", 9, False),
    ("CML-07     Elbow downstream of P-101  9.5        6.1        BELOW t-min", 9, True),
    ("CML-08     Straight run C             9.5        8.0        Satisfactory", 9, False),
    ("CML-09     Branch connection          9.5        7.6        Monitor", 9, False),
    ("CML-10     Support location           9.5        8.3        Satisfactory", 9, False),
    ("CML-11     Vertical riser             9.5        8.6        Satisfactory", 9, False),
    ("CML-12     Dead leg near drain        9.5        6.6        Monitor closely", 9, False),
    ("", 8, False),
    ("@@PAGEBREAK", 0, False),
    ("3. OBSERVATIONS", 10, True),
    ("3.1 Location CML-07 recorded a measured thickness of 6.1 mm which is below "
     "the minimum required thickness of 6.4 mm. The location is an elbow "
     "immediately downstream of pump P-101A discharge and shows evidence of "
     "flow accelerated corrosion.", 9, False),
    ("3.2 Location CML-12 at the dead leg near the drain valve recorded 6.6 mm "
     "against t-min of 6.4 mm. Remaining margin is 0.2 mm only.", 9, False),
    ("3.3 External coating degradation was observed at the pipe rack crossing "
     "between CML-05 and CML-06. Localised corrosion under insulation cannot be "
     "ruled out at this location.", 9, False),
    ("3.4 No leakage or through wall defect was observed during the survey.", 9, False),
    ("", 8, False),
    ("4. RECOMMENDATIONS", 10, True),
    ("4.1 Line 12in-P-0421 shall not be operated above 14.0 kg/cm2 until the "
     "CML-07 location is repaired or replaced.", 9, False),
    ("4.2 A deviation approval is required from DGM (Mechanical) for continued "
     "operation as per clause 7.2 of SOP-MEC-014.", 9, False),
    ("4.3 Spool replacement of approximately 2.5 metres covering CML-07 is "
     "recommended during the next available opportunity. Estimated cost "
     "Rs. 4.85 lakh.", 9, False),
    ("4.4 Re-inspection interval for CML-12 to be reduced from 36 months to "
     "12 months.", 9, False),
    ("", 8, False),
    ("Prepared by: R. Krishnan          Reviewed by: S. Deshmukh, Sr. Manager", 9, False),
    ("(SYNTHETIC DOCUMENT - CREATED FOR SIH PROTOTYPE DEMONSTRATION)", 7, False),
]

SOP = [
    ("SOP-MEC-014  (SYNTHETIC DEMO DOCUMENT)", 12, True),
    ("STANDARD OPERATING PROCEDURE - THICKNESS MONITORING AND DEVIATION APPROVAL", 11, True),
    ("Revision R3   Effective 01.04.2026   Owner: Inspection Department", 9, False),
    ("", 8, False),
    ("7. RETIREMENT THICKNESS AND DEVIATIONS", 10, True),
    ("7.1 Where the measured thickness at any condition monitoring location is "
     "found to be equal to or greater than the minimum required thickness "
     "(t-min), the equipment may continue in service at design conditions and "
     "the normal inspection interval shall apply.", 9, False),
    ("7.2 Where the measured thickness at any location is found to be below "
     "t-min, continued operation shall require a written deviation approval. "
     "The deviation note shall be prepared by the Inspection Department, "
     "recommended by the Sr. Manager (Inspection) and approved by the DGM "
     "(Mechanical). The note shall state the reduced operating limit, the "
     "compensatory measures and the target date for permanent repair.", 9, False),
    ("7.3 Where remaining life computed from the measured thickness and the "
     "established corrosion rate is less than 24 months, the inspection "
     "interval for that location shall be reduced to not more than 12 months.", 9, False),
    ("7.4 A deviation approval under clause 7.2 shall remain valid for a maximum "
     "period of 180 days from the date of approval or until the next scheduled "
     "shutdown, whichever is earlier.", 9, False),
    ("", 8, False),
    ("8. REMAINING LIFE CALCULATION", 10, True),
    ("8.1 Remaining life in years shall be computed as the measured thickness "
     "minus the minimum required thickness, divided by the established "
     "corrosion rate in mm per year.", 9, False),
    ("8.2 Where the computed remaining life is zero or negative, the location "
     "shall be treated as having reached end of life and clause 7.2 shall "
     "apply immediately.", 9, False),
    ("", 8, False),
    ("9. HOT WORK DURING REPAIR", 10, True),
    ("9.1 A hot work permit issued for repair activity shall be valid for one "
     "shift of 8 hours and shall be revalidated by the area authority before "
     "continuation into the next shift.", 9, False),
    ("(SYNTHETIC DOCUMENT - CREATED FOR SIH PROTOTYPE DEMONSTRATION)", 7, False),
]

DOA = [
    ("DELEGATION OF AUTHORITY  (SYNTHETIC DEMO DOCUMENT)", 12, True),
    ("FINANCE AND CONTRACTS MANUAL - SCHEDULE II", 10, True),
    ("Revision R7   Effective 01.04.2026", 9, False),
    ("", 8, False),
    ("2. CAPITAL AND REVENUE EXPENDITURE LIMITS", 10, True),
    ("2.1 Manager may approve expenditure up to Rs. 2.00 lakh per case.", 9, False),
    ("2.2 Senior Manager may approve expenditure up to Rs. 5.00 lakh per case.", 9, False),
    ("2.3 Deputy General Manager may approve expenditure up to Rs. 25.00 lakh "
     "per case for maintenance and repair items.", 9, False),
    ("2.4 General Manager may approve expenditure up to Rs. 100.00 lakh per case.", 9, False),
    ("2.5 Any proposal exceeding Rs. 100.00 lakh shall be placed before the "
     "Director (Operations).", 9, False),
    ("", 8, False),
    ("3. EMERGENCY PROCUREMENT", 10, True),
    ("3.1 In case of a plant emergency, the Deputy General Manager may approve "
     "emergency procurement up to Rs. 10.00 lakh subject to ratification within "
     "15 working days.", 9, False),
    ("(SYNTHETIC DOCUMENT - CREATED FOR SIH PROTOTYPE DEMONSTRATION)", 7, False),
]

SHUTDOWN = [
    ("UNIT-2 SHUTDOWN COMPLETION REPORT  (SYNTHETIC DEMO DOCUMENT)", 12, True),
    ("Turnaround TA-2026-01   Duration: 21 days   Completed 18.03.2026", 9, False),
    ("", 8, False),
    ("1. SUMMARY", 10, True),
    ("The planned turnaround of Unit-2 Crude Distillation was completed in 21 "
     "days against a plan of 19 days. The overrun of 2 days was attributed to "
     "additional tube plugging in exchanger E-204 and delayed arrival of the "
     "reactor internals from the vendor.", 9, False),
    ("2. MAJOR JOBS COMPLETED", 10, True),
    ("2.1 Column C-201 tray inspection and replacement of 14 damaged trays.", 9, False),
    ("2.2 Exchanger E-204 retubing, 312 tubes replaced against a plan of 280.", 9, False),
    ("2.3 Furnace F-201 tube replacement in the radiant section, 22 tubes.", 9, False),
    ("2.4 Replacement of 3 PSVs, namely PSV-2101, PSV-2104 and PSV-2118.", 9, False),
    ("3. PENDING ITEMS", 10, True),
    ("3.1 Insulation restoration at the pipe rack crossing is pending and has "
     "been carried forward to the next opportunity.", 9, False),
    ("3.2 Spool replacement on line 12in-P-0421 could not be taken up as the "
     "material was not available. Carried forward.", 9, False),
    ("4. COST", 10, True),
    ("4.1 Total turnaround expenditure was Rs. 1420.00 lakh against a sanctioned "
     "budget of Rs. 1350.00 lakh, an overrun of 5.2 percent.", 9, False),
    ("(SYNTHETIC DOCUMENT - CREATED FOR SIH PROTOTYPE DEMONSTRATION)", 7, False),
]


def main():
    print("building synthetic demo corpus ...")
    ut_pdf = make_pdf("_ut_source.pdf", UT_REPORT)
    spdf = scan_pdf(ut_pdf, OUT / "UT-Thickness-Report-U2-0417_SCAN.pdf")
    png = scan_effect(ut_pdf, OUT / "UT-Thickness-Report-U2-0417_page1.png")
    ut_pdf.unlink(missing_ok=True)
    print(f"  scanned pdf   : {spdf.name} (image-only, no text layer)")
    print(f"  scanned image : {png.name}")

    for name, body in (("SOP-MEC-014-Thickness-Monitoring.pdf", SOP),
                       ("Delegation-of-Authority-Schedule-II.pdf", DOA),
                       ("Unit-2-Shutdown-Completion-Report.pdf", SHUTDOWN)):
        p = make_pdf(name, body)
        print(f"  kb document   : {p.name}")

    # ingest the three text-layer PDFs into the knowledge base
    sys.path.insert(0, str(ROOT))
    from backend import kb as KB
    for p in sorted(OUT.glob("*.pdf")):
        if "SCAN" in p.name:
            continue          # the scan is the demo INPUT, not pre-loaded knowledge
        info = KB.ingest_file(p, title=p.stem, doc_type="sop")
        print(f"  ingested      : {p.name} -> {info['chunks']} chunks "
              f"({info.get('source')})")
    print("  kb stats      :", KB.stats())
    print("done.")


if __name__ == "__main__":
    main()
