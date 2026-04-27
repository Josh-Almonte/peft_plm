# Data

Place the sub-cellular localization dataset at:

- `data/raw/subcellular_localization.csv`

Required columns:

- `sequence`: amino-acid sequence string.
- `label`: one of the 10 localization classes.
- `split`: one of `train`, `val`, `test`.

Example row:

```csv
sequence,label,split
MNNIRRVV...,Nucleus,train
```

Notes:
- Sequences are automatically uppercased and tokenized as space-delimited residues for ProT5.
- Rare amino acids `U`, `Z`, `O`, `B` are mapped to `X` in preprocessing.
