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
python experiments/{svm, lda, logreg}_experiment.py --dataset_name {"VEPESS", "BCICa", "BCICb"} --data_path "path_to_dataset"
```

To perform binary classification on the BCICa dataset, also add the `--two-class` flat to the command. 


