# MStoCIRC2

`MStoCIRC2` is a command-line proteogenomics toolkit for discovering, validating, and quantifying protein-coding circular RNAs from mass-spectrometry data.

## What It Does

- `mstocirc2 orf`: predict circRNA ORFs and export circRNA-aware FASTA and mapping files
- `mstocirc2 search`: assemble a circRNA-aware search database and launch FragPipe workflows
- `mstocirc2 eval`: evaluate and rank circRNA translation potential using IRES, m6A, and peptide-support evidence
- `mstocirc2 dea`: perform quantitative filtering, protein roll-up, differential expression analysis (DEA), correlation analysis, and enrichment for circRNA-derived candidates
- `mstocirc2 nonquant`: run `orf -> search -> eval` in one command
- `mstocirc2 quant`: run `orf -> search -> eval -> dea` in one command

## Installation

### Recommended: conda environment from the repository root

```bash
git clone https://github.com/mikrokosmosV/MStoCIRC2.git
cd MStoCIRC2
conda env create -f conda_env.yaml
conda activate mstocirc2
mstocirc2 --help
```

This is the simplest installation path for most users. `conda_env.yaml` installs:

- the tested Python runtime for `MStoCIRC2`
- the core Python packages listed in `requirements.txt`
- the R runtime used by `mstocirc2 dea`
- core command-line tools used directly by the pipeline, including `bedtools`, `diamond`, and `openjdk`

### Pip installation into an existing environment

```bash
pip install -r requirements.txt
pip install -e .
mstocirc2 --help
```

Use the pip route only if you already manage your own Python environment. It installs the Python package layer only. External tools such as `bedtools`, `diamond`, FragPipe, DIA-NN, and the optional R runtime must already be available separately.

## Runtime Requirements

- Core runtime dependencies are centralized in `conda_env.yaml`, `requirements.txt`, and the bundled DEA helper installers. Representative components include `diamond` for sequence/background filtering, `torch` for bundled `DeepCircM6A` inference, and the R package stack used by `mstocirc2 dea` for differential analysis and downstream enrichment.
- External tools such as FragPipe, DIA workflow Python packages, and DeepCIP must be installed and managed by the user. They do not need to live in the same environment as `mstocirc2`, because the CLI can call site-specific executables, tool directories, and Python interpreters through command-line parameters.

### FragPipe runtime

`MStoCIRC2` uses [FragPipe](https://fragpipe.nesvilab.org/) for the `search`, `nonquant`, and `quant` stages.

Important note:

- Before running FragPipe, users must manually download the academic-license
  tools [MSFragger](https://msfragger.nesvilab.org/),
  [IonQuant](https://ionquant.nesvilab.org/), and
  [diaTracer](https://diatracer.nesvilab.org/) and place them under
  `src/mstocirc2/fragpipe/tools/` or the site-specific FragPipe tools directory
  supplied through `--tools-dir`

### DIA-specific Python runtime

For DIA workflows, the FragPipe Python selected by `-py/--python-bin` or the current `mstocirc2` interpreter must already provide:

- `fragpipe-speclib`
- `easypqp`
- `lxml`

These packages must be installed by the user in the Python environment that FragPipe will actually use. That interpreter can be different from the one used to run `mstocirc2`.

### Translation-potential scoring

- `DeepCIP` for IRES-related scoring
- bundled `DeepCircM6A` inference assets, which require `torch` in the Python environment running `mstocirc2 eval`

`DeepCIP` is an external dependency and must be installed by the user. If it is unavailable, the main workflow still runs, but translation-potential scoring in `circ_predict.txt` is less complete.

`MStoCIRC2` does not require DeepCIP to share the same Python environment as the main CLI. If your DeepCIP installation only works with a separate Python, pass that interpreter through `--deepcip-python`.

DeepCIP upstream resources:

- code repository: [DeepCIP](https://github.com/zjupgx/DeepCIP.git)
- ViennaRNA official repository: [ViennaRNA](https://github.com/ViennaRNA/ViennaRNA)
- Seqkit download page: [Seqkit](https://bioinf.shenwei.me/seqkit/download/)

For the full upstream installation guide, use the official DeepCIP repository directly.

## Quick Start

### End-to-end workflow

```bash
mstocirc2 nonquant \
  -cs examples/input/circRNA_sequences.fasta \
  -cp examples/input/canonical_protein.fasta \
  -mf manifest.fp-manifest \
  -py python \
  -o results_nonquant
```

```bash
mstocirc2 quant \
  -cs examples/input/circRNA_sequences.fasta \
  -cp examples/input/canonical_protein.fasta \
  -mf manifest.fp-manifest \
  -py python \
  -o results_quant
```

### Stage-by-stage example

```bash
mstocirc2 orf \
  -cs examples/input/circRNA_sequences.fasta \
  -cp examples/input/canonical_protein.fasta
```

```bash
mstocirc2 search \
  -cs examples/input/circRNA_sequences.fasta \
  -cp examples/input/canonical_protein.fasta \
  -mf /path/to/manifest.fp-manifest \
  -fb /path/to/fragpipe/bin/fragpipe
```

```bash
mstocirc2 eval \
  -fi examples/output/orf \
  -ms examples/input/results_search \
  -cp /path/to/canonical_proteins.fasta
```

```bash
mstocirc2 dea \
  -pm examples/dea/peptide_matrix.tsv \
  -cr examples/dea/circrna_reference.tsv \
  -de examples/dea/design.txt \
  -st generic
```

**Note:** The corresponding example outputs generated during our testing can be found in the `examples/output` directory for your reference.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
