# GPT-Medic

A foundation model for learning patient health trajectories from electronic health records, trained on the [MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/2.2/) dataset.

## Overview

GPT-Medic tokenizes structured EHR data into a unified vocabulary and trains an autoregressive transformer to model sequential patient health events.

### What it does

- Converts clinical data (diagnoses, procedures, medications, labs, vitals) into tokens
- Learns temporal patterns between medical events
- Generates synthetic patient trajectories

### Tokenization

| Data Type | Source | Method |
|-----------|--------|--------|
| Demographics | patients, admissions | Categorical tokens |
| Diagnoses | diagnoses_icd | Hierarchical ICD-10 decomposition |
| Procedures | procedures_icd | ICD-10-PCS character tokens |
| Medications | prescriptions | ATC code hierarchy |
| Lab Results | labevents | Lab name + quantile (Q1-Q10) |
| Time Gaps | computed | Bucketed intervals (5min to 6months) |

### Model

- **Architecture**: 4-layer GPT-style transformer
- **Parameters**: 2.5M
- **Context**: 256 tokens
- **Validation Perplexity**: 7.64

### Example Output

```
SEX_F MARITAL_SINGLE RACE_WHITE <ICD_F_I10> <ICD_F_E11> <TIME_2h-6h>
<LAB Hemoglobin> <Q5> <LAB Glucose> <Q8> ATC_A10 ATC_SUFFIX_B01
```

## Quick Start

```bash
uv sync
```

```python
from data_loader import load_mimic_tables
from tokenizer.timeline import PatientTimelineTokenizer

tables = load_mimic_tables("../data/")
tokenizer = PatientTimelineTokenizer(n_quantiles=10)
tokenizer.fit(tables)

result = tokenizer.tokenize_session(hadm_id=28258130, tables=tables)
```

See `explore.ipynb` for training.
