import os
from typing import Literal, Union, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class OddOneOutSignalDataLoader:
    def __init__(
        self,
        data_dir: str,
        label_file: str,
    ):
        self.data_dir: str = data_dir
        self.label_file: str = label_file

    @staticmethod
    def standardize_subjectwise(dataset: pd.DataFrame) -> pd.DataFrame:
        # Ensure a copy of the dataset is used to avoid SettingWithCopyWarning
        dataset = dataset.copy()

        for subject in pd.unique(dataset["Subject"]):
            subject_filter = dataset["Subject"] == subject

            # Stack the data for the subject into a 3D array
            subject_data_stacked = np.stack(
                dataset.loc[subject_filter, "Data"].to_numpy(), axis=0
            )

            subject_data_stacked = (
                subject_data_stacked
                - subject_data_stacked.mean(axis=(0, 2), keepdims=True)
            ) / subject_data_stacked.std(axis=(0, 2), keepdims=True)

            # Assign reshaped data back to the DataFrame safely as a Series
            reshaped_data_series = pd.Series(
                [sample for sample in subject_data_stacked],
                index=dataset.loc[subject_filter].index,
            )
            dataset.loc[subject_filter, "Data"] = reshaped_data_series

        return dataset

    def get_folds(
        self,
        test_subjects: Union[str, list, int],
        standardize_by: Optional[Literal["sample", "subject", "set"]] = None,
        encode_labels: Optional[Literal["label", "onehot"]] = None,
        return_subjects: bool = False,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        # Load the dataset from the label file
        dataset = pd.read_csv(self.label_file, sep=";")

        # Determine test subjects and create a mask for selection
        if isinstance(test_subjects, str) or isinstance(test_subjects, int):
            test_subject_selection = dataset["Subject"] == test_subjects
        elif isinstance(test_subjects, list):
            test_subject_selection = dataset["Subject"].isin(test_subjects)
        else:
            raise AttributeError("Wrong type for test_subjects. Must be str or list.")

        # Extract train and test labels
        train_labels = dataset.loc[~test_subject_selection, "Label"].to_numpy()
        test_labels = dataset.loc[test_subject_selection, "Label"].to_numpy()

        # Encode labels if specified (left right feat tongue)
        if encode_labels:
            if encode_labels == "label":
                le = LabelEncoder()
                train_labels = le.fit_transform(train_labels)
                test_labels = le.transform(test_labels)
            elif encode_labels == "onehot":
                ohe = OneHotEncoder(sparse_output=False)  # Updated for compatibility
                train_labels = ohe.fit_transform(train_labels.values.reshape(-1, 1))
                test_labels = ohe.transform(test_labels.values.reshape(-1, 1))

        # Create copies of train and test sets to avoid SettingWithCopyWarning
        train_set_data = dataset.loc[~test_subject_selection].copy()
        test_set_data = dataset.loc[test_subject_selection].copy()

        # Populate the 'Data' column with loaded signals from .mat files
        train_set_data["Data"] = train_set_data["Path"].apply(
            lambda x: loadmat(os.path.join(self.data_dir, x))["Samples"]
        )
        test_set_data["Data"] = test_set_data["Path"].apply(
            lambda x: loadmat(os.path.join(self.data_dir, x))["Samples"]
        )

        # Standardization if requested
        if standardize_by:
            if standardize_by == "sample":
                train_set_data["Data"] = train_set_data["Data"].apply(
                    lambda x: (x - x.mean(axis=-1, keepdims=True))
                    / x.std(axis=-1, keepdims=True)
                )
                test_set_data["Data"] = test_set_data["Data"].apply(
                    lambda x: (x - x.mean(axis=-1, keepdims=True))
                    / x.std(axis=-1, keepdims=True)
                )
            elif standardize_by == "subject":
                train_set_data = OddOneOutSignalDataLoader.standardize_subjectwise(
                    train_set_data
                )
                test_set_data = OddOneOutSignalDataLoader.standardize_subjectwise(
                    test_set_data
                )
            elif standardize_by == "set":
                all_train_samples = np.stack(train_set_data["Data"].to_numpy())
                train_features_mean = np.mean(
                    all_train_samples, axis=(0, 2), keepdims=True
                ).squeeze(0)
                train_features_std = np.std(
                    all_train_samples, axis=(0, 2), keepdims=True
                ).squeeze(0)
                train_set_data["Data"] = train_set_data["Data"].apply(
                    lambda x: (x - train_features_mean) / train_features_std
                )

                # all_test_samples = np.stack(test_set_data["Data"].to_numpy())
                # test_features_std, test_features_mean = all_test_samples.std_mean(
                #     axis=(0, 2), keepdims=True
                # )
                test_set_data["Data"] = test_set_data["Data"].apply(
                    lambda x: (x - train_features_mean) / train_features_std
                )
            else:
                raise AttributeError(
                    "Wrong value for standardize_by. Must be 'sample', 'subject' or 'set'."
                )

        if return_subjects:
            return (
                train_set_data["Data"],
                train_labels,
                test_set_data["Data"],
                test_labels,
                train_set_data["Subject"].to_numpy(),
                test_set_data["Subject"].to_numpy(),
            )
        else:
            return (
                train_set_data["Data"],
                train_labels,
                test_set_data["Data"],
                test_labels,
            )
