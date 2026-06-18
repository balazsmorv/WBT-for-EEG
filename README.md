# WBT-for-EEG
This repository contains source code for the article `Enhancing multi-paradigm EEG signal classification in cross-subject settings using optimal transport`.

## Prerequisites
Running the experiments require some Python packages (requirements.txt), including MLFlow, where all the results, metrics and figures are saved to. To enable MLFlow, set a tracking URI and start an MLFlow server on that URI, like 

``` Python
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# To setup an mlflow server: mlflow server --host 127.0.0.1 --port 8080
```

## Reproducing results
To reproduce our results, run the corresponding experiments, like: 

``` shell
python {svm, lda, logreg, nn}_experiment.py --dataset_name {"VEPESS", "BCICa", "BCICb"} --data_path "path_to_dataset"
```

To perform binary classification on the BCICa dataset, in each experiment, remove the commenting #-s in the code where we filter for binary labels.

## Cite the article
If you found this repository useful, please cite the original article:

```
@article{MORVAY2026108892,
  title = {Enhancing multi-paradigm EEG signal classification in cross-subject settings using optimal transport},
  journal = {Biomedical Signal Processing and Control},
  volume = {113},
  pages = {108892},
  year = {2026},
  issn = {1746-8094},
  doi = {https://doi.org/10.1016/j.bspc.2025.108892},
  url = {https://www.sciencedirect.com/science/article/pii/S174680942501403X},
  author = {Balázs Tibor Morvay and Szabolcs Torma and József Pitrik and Luca Szegletes},
  keywords = {Electroencephalography, Cross-subject classification, Machine learning, Optimal transport, Domain adaptation},
}
```
