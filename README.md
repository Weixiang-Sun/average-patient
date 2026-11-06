<h1 align="center">The Average Patient</h1>
<h3 align="center">Do Simulated Psychiatric Patients Answer Like Real People?</h3>

<p align="center">
  <a href="https://weixiang-sun.github.io/">Weixiang Sun</a><sup>1</sup>,
  <a href="https://zehong-wang.github.io/">Zehong Wang</a><sup>1†</sup>,
  <a href="https://shiiiikw.github.io/">Kaiwang Shi</a><sup>1</sup>,
  <a href="https://hongh0.github.io/">Hong Huang</a><sup>2</sup>,
  <a href="https://www.linkedin.com/in/colby-p-nelson/">Colby Nelson</a><sup>1</sup>,
  <a href="https://yiyang-ian-li.github.io/">Yiyang Li</a><sup>1</sup>,
  <a href="http://yes-lab.org/">Yanfang Ye</a><sup>1†</sup>
</p>
<p align="center"><sup>1</sup>University of Notre Dame &nbsp;&nbsp; <sup>2</sup>Simon Fraser University &nbsp;&nbsp; <sup>†</sup>Corresponding authors</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://weixiang-sun.github.io/average-patient/"><img src="https://img.shields.io/badge/Project%20Page-average--patient-D97757?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://github.com/Weixiang-Sun/average-patient"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white" alt="Code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2ecc71" alt="License: MIT"></a>
</p>

## Overview

Language models are used as simulated psychiatric patients to train clinicians and to evaluate AI therapists and assessment agents, and they are usually judged by how plausible their transcripts look or by their total scores. We compare simulated patients with real people who report the same total score, item by item, across 17 models and nine human references, and with people instructed to feign a disorder. The [project page](https://weixiang-sun.github.io/average-patient/) explains the comparison with interactive figures.

- Simulated patients vary 1.5 to 11 times less than real respondents with the same total and endorse nearly every symptom at a middling level.
- Prompt changes, published simulator personas, larger models and reasoning reduce the pattern without removing it.
- In clinical interviews, uniform patients leave an assessment agent's recognition of absent symptoms untested; having the model word a real symptom profile restores much of the variation.

## Setup

```
uv sync
```

Model runs read API credentials from a `.env` file (`AZURE_OPENAI_ENDPOINT`, `AZURE_ANTHROPIC_ENDPOINT`, `AZURE_ANTHROPIC_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`).

## Data

The human reference data are public and are not included. Download them and place them under `data/`:

| data | source |
|---|---|
| NHANES PHQ-9, 2005-2023 | [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/) (downloaded by `scripts/01_download_nhanes.py`) |
| NHIS 2016-2019 (K6, GAD-7, PHQ-8) | [CDC NHIS](https://www.cdc.gov/nchs/nhis/) |
| PCL-5 instructed-feigning study | [OSF 6b8jf](https://osf.io/6b8jf) |
| C19PRC-UK (OCI-R, PHQ-15, ITQ) | [OSF v2zur](https://osf.io/v2zur/) |
| EDE-Q, Mexican general population | [Contreras-Valdez et al., 2022](https://doi.org/10.1371/journal.pone.0266507) |
| PCL-5, Greek women after caesarean section | [Orovou et al., 2021](https://doi.org/10.1371/journal.pone.0255689) |
| COACH baseline PHQ-9 | [Harvard Dataverse](https://doi.org/10.7910/DVN/6RWH41) |
| TalkDep personas | [Wang et al., 2025](https://github.com/Anxo06/TalkDep) |
| CACTUS intake forms | [Lee et al., 2024](https://github.com/LangAGI-Lab/cactus) |

## Reproducing the results

```
# human references and frozen norms
uv run python scripts/01_download_nhanes.py
uv run python -c "import sys; sys.path.insert(0, 'src'); from mental_act import nhis, extra; nhis.build(); extra.build_human()"
uv run python scripts/02_build_norms.py && uv run python scripts/10_build_norms_generic.py && uv run python scripts/06_build_prototype_pole.py

# model runs: the launch scripts in scripts/*.sh list every command

# every analysis, the paper's tables and the number check
bash scripts/99_rerun_analyses.sh
```

Model responses and the human ratings of interview transcripts are not released.

## Citation

Coming soon.

## License

MIT. See [LICENSE](LICENSE).
