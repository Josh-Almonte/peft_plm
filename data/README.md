# Data

Download the raw dataset from [HannesStark/protein-localization](https://github.com/HannesStark/protein-localization). You need the three split files: `train.pkl`, `valid.pkl`, `test.pkl`.

Then run `data/data_preprocess_to_csv.ipynb` (updating the input paths to your local copies) to produce:

- `data/raw/subcellular_localization.csv`

Required columns:

- `sequence`: amino-acid sequence string.
- `label`: one of the 10 localization classes.
- `split`: one of `train`, `valid`, `test`.

Example row:

```csv
sequence,label,split
MNNIRRVV...,Nucleus,train
```

Notes:
- Sequences are automatically uppercased and tokenized as space-delimited residues for ProT5.
- Rare amino acids `U`, `Z`, `O`, `B` are mapped to `X` in preprocessing.
