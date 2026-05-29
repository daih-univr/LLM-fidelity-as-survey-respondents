# LLM-fidelity-as-survey-respondents

Code, data, and analysis outputs for the manuscript:

**under review**

This repository contains the material used to compare human survey responses and LLM-generated survey responses across three response formats:

- open-ended responses
- one-word associations
- Likert-type items

The repository includes:
- scripts for generating synthetic survey responses from multiple LLMs
- scripts for feature extraction and sentiment analysis
- scripts for fidelity analysis
- exported intermediate and final analysis outputs


## ⚠️️️⚠️️️⚠️️️ Disclaimer on generated content ⚠️️️⚠️️️⚠️️️

The files in `generated-surveys/` are model outputs as generated in the experimental pipeline, without manual curation of individual responses. Because these texts were produced automatically by large language models, they may contain harmful, offensive, biased, misleading, or otherwise objectionable content. They are included solely for transparency and reproducibility of the research. The views, wording, and claims expressed in those generated responses are attributable to the models and should not be interpreted as reflecting the views of the authors.


## Repository structure

### `code/`

Python scripts used for the generation pipeline and analysis

- `simulateXYZ.py`  
  Generates synthetic survey responses using a XYZ model.

- `extractFeatures.py`  
  Extracts structured features from raw survey responses for downstream analysis.

- `computeSentiment.py`  
  Computes sentiment-related variables for open-ended and/or word-association responses via Gemma 3 27B

- `fidelityAnalysis.py`  
  Runs the main fidelity analyses comparing human and model-generated responses.

### `generated-surveys/`

Model-generated survey outputs, organized by model family.

Each subfolder contains the outputs used in the analyses. For example, the `gemma/` folder includes:
- filled surveys in a single spreadsheet export
- sentiment-specific JSON files

### `analysis_outputs/`

Processed outputs used to generate the manuscript tables, figures, and summary statistics.

Main files include:

- `open_embedding_fidelity.csv`  
  Open-response semantic fidelity metrics, including embedding-based distance measures.

- `open_sentiment.csv`  
  Sentiment comparison results for open-ended responses.

- `wordassoc_energy_distance.csv`  
  Energy-distance results for one-word association responses.

- `wordassoc_entropy_repetition_unique.csv`  
  Lexical diversity, repetition, and uniqueness indicators for one-word association data.

- `wordassoc_sentiment_shift.csv`  
  Sentiment-shift results for one-word association responses.

- `likert_props.csv`  
  Proportional response distributions for Likert items.

- `likert_chi2.csv`  
  Chi-square comparison results for Likert distributions.

- `skepticism_diff_vs_human.csv`  
  Model-versus-human differences for skepticism index.

- `top10_open.csv`  
  Top distinctive lexical items or themes in open-ended responses.

- `top10_wordassoc.csv`  
  Top distinctive one-word associations.

- `plots/`  
  Exported plots and tables used in the manuscript.

## Analytical workflow

The overall workflow is:

1. Generate synthetic survey responses for each model.
2. Extract linguistic and distributional features.
3. Compute sentiment-based indicators.
4. Enrich and structure the generated outputs.
5. Compare LLM and human responses across:
   - semantic distance
   - sentiment shift
   - lexical diversity and repetition
   - ordinal response distributions

## Expected inputs

The scripts assume access to:
- the original human survey data (not provided)
- model-generated survey outputs
- API credentials or local inference setup for the models used


## Environment

This project uses Python.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes on model access

Some generation scripts may require:
- API-based access to proprietary models
- manual configuration of authentication keys
- local inference for open-weight models


## License

```text
Code is released under the MIT License.
Data files are shared for research use only.
```