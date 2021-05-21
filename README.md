# Cross-Lingual Offensive Language Identification

Authors: Nikolina Grabovica, Selma Halilčević, Matjaž Mav

Advisors: Slavko Žitnik

Organization: University of Ljubljana, Faculty of Computer and Information Science

Course: Natural Language Processing 2020/2021

---

## Description

TODO - Quick description

---

## Requirements
- [Conda](https://docs.conda.io/en/latest/miniconda.html)

## Installation
- `$ conda create --name nlp --file requirements.txt`
- `$ conda activate nlp`
- Make sure that jupyter notebooks are run with repository root as working directory
- Download trained model checkpoints from here (12GB): https://drive.google.com/file/d/10r0ixTeOgG1AxDBksGMPjsUsO_sLe9gD/view?usp=sharing
- Place checkpoints into repository root, see folder structure for details

## Folder structure
```txt
├── .gitignore                      Git ignore config
├── README.md                       This file
├── requirements.txt                Conda environment definition
├── data/                           Contains datasets 
├── reports/                        Contains reports
├── results/                        Contains final results and visualizations
├── checkpoints/                    !!Contains downloaded checkpoints, see installation steps!!
    ├── elmoformanylanguages/       Contains pre-trained ELMo for EN and SI language
    ├── outputs/                    Contains pre-trained BERT, mBERT, T5 and mT5 models
    ├── .gitignore                  
└── src/                            Contains source files
    └── eval-*.ipynb                Model evaluation notebooks


```

## Links
- Overleaf report: https://www.overleaf.com/read/jypncsrbgmmc
