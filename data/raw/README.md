# data/raw — real cohort sources (gitignored, re-acquire from upstream)

This directory holds raw clinical-cohort data downloaded from public sources.
The actual files are gitignored (size + author courtesy); this README documents
the upstream URLs so anyone cloning the repo can re-acquire.

## Bruchovsky et al. IADT cohort (the "dataTanaka" archive)

**Upstream:** http://www.nicholasbruchovsky.com/dataTanaka.zip
**Cohort:** 72 patients on intermittent androgen-deprivation therapy (Bruchovsky et al. 2006 / 2007). The same cohort fit by Brady-Nicholls et al. 2020 (Nat Comms), Strobl et al. 2022 (Comms Med), and Gallagher et al. 2025 (bioRxiv).
**Acquisition (one-liner):**

```bash
mkdir -p data/raw
cd data/raw
curl -sLO http://www.nicholasbruchovsky.com/dataTanaka.zip
unzip -q dataTanaka.zip
```

Result: `data/raw/dataTanaka/{Bruchovsky_et_al,Shaw_et_al,classificationsofpatients.txt,readme.txt}` with ~72 patient files in `Bruchovsky_et_al/patient*.txt`.

**File schema** (per `dataTanaka/readme.txt`):
- col 1: patient_id (int)
- col 2: date (YYYY/M/D HH:MM:SS)
- col 3: CPA dose (mg)
- col 4: LEU dose (mg)
- col 5: PSA (ng/mL) — **primary observation channel**
- col 6: testosterone (nmol/L)
- col 7: cycle number
- col 8: treatment (1=on, 0=off) — used as `u_schedule`
- col 9: day number (relative)
- col 10: day number (alternate origin) — used as `t_obs`

**Loaded via:** `realdata.load_dataTanaka()` (see `src/realdata/bruchovsky.py`).

**Citation:**
- Bruchovsky N, Klotz L, Crook J, Phillips N, Abersbach J, Goldenberg SL. *Quality of life, morbidity, and mortality results of a prospective phase II study of intermittent androgen suppression for men with evidence of prostate-specific antigen relapse after radiation therapy for locally advanced prostate cancer.* Clinical Genitourinary Cancer 6(1):46-52 (2008). doi:10.3816/CGC.2008.n.008.
- Tanaka G, Hirata Y, Goldenberg SL, Bruchovsky N, Aihara K. *Mathematical modelling of prostate cancer growth and its application to hormone therapy.* Phil. Trans. R. Soc. A 368:5029-5044 (2010). doi:10.1098/rsta.2010.0221.

The acquisition is open access; no IRB / DUA needed beyond standard academic-courtesy citation.

## Brady-Nicholls 2020 supporting code (NOT raw data, just MATLAB scripts)

**Upstream:** https://github.com/reneebrady/IADT_PCaSC (archived 2020-02-14)
**Acquisition:** `curl -sLO https://github.com/reneebrady/IADT_PCaSC/archive/refs/heads/master.zip` then unzip + extract `Archive.zip`. Contains MATLAB code only; raw cohort data is at the Bruchovsky URL above.

## Future / aspirational

Zhang 2017 (mCRPC adaptive-therapy pilot trial) — supplementary data is in the
Nature Communications supplementary PDF; would require manual extraction from
PDF tables. Not yet automated.
