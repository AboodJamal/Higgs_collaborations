# ------------------------------
# Dummy Sample Submission
# ------------------------------

BDT = True
NN = False

from statistical_analysis import calculate_saved_info, compute_mu
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")
from systematic_analysis import SystModel
import logging

logging.basicConfig(
    level=logging.INFO,  # Set minimum level to INFO (so DEBUG is ignored)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Write logs to file
        logging.StreamHandler(),  # Also print logs to console
    ],
)
logger = logging.getLogger(__name__)
from systematic_analysis import SystModel


# Dummy preselection logic (does nothing for now)
class DummyPreselection:
    def apply_pre_selection(self, dataset, threshold=0.8):
        return dataset


# Main class for training and prediction. Used by the competition ingestion system.
class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init :
        takes 3 arguments: train_set systematics and model_type.
        can be used for initializing variables, classifier etc.
    2) fit :
        takes no arguments
        can be used to train a classifier
    3) predict:
        takes 1 argument: test sets
        can be used to get predictions of the test set.
        returns a dictionary

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods

            When you add another file with the submission model e.g. a trained model to be loaded and used,
            load it in the following way:

            # get to the model directory (your submission directory)
            model_dir = os.path.dirname(os.path.abspath(__file__))

            your trained model file is now in model_dir, you can load it from here
    """

    def __init__(self, get_train_set=None, systematics=None, model_type="sample_model"):
        """
        Initializes model by loading and splitting data,
        selecting the model type (NN, BDT, or SampleModel),
        and preparing systematics handler.
        """

        indices = np.arange(1400_000)

        np.random.shuffle(indices)

        train_indices = indices[:500_000]
        holdout_indices = indices[500_000:900_000]
        valid_indices = indices[900_000:]

        # Load and split training data
        training_df = get_train_set(selected_indices=train_indices)
        """
        {
        "labels": pd.Series (binary target),
        "weights": pd.Series (float sample weights),
        "detailed_labels": pd.Series (str category like 'ttbar'),
        "data": pd.DataFrame (28 numerical features)
        }

        """
        self.training_set = {
            "labels": training_df.pop("labels"),
            "weights": training_df.pop("weights"),
            "detailed_labels": training_df.pop("detailed_labels"),
            "data": training_df,
        }

        # Flag indicating whether the model has been trained yet.
        self.istrained = False
        self.preselection = DummyPreselection()

        # Frees up memory — training_df is no longer needed after its contents were split and stored.
        del training_df

        self.systematics = systematics

        print("Training Data: ", self.training_set["data"].shape)
        print("Training Labels: ", self.training_set["labels"].shape)
        print("Training Weights: ", self.training_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 0].sum(),
        )

        valid_df = get_train_set(selected_indices=valid_indices)

        self.valid_set = {
            "labels": valid_df.pop("labels"),
            "weights": valid_df.pop("weights"),
            "detailed_labels": valid_df.pop("detailed_labels"),
            "data": valid_df,
        }
        del valid_df

        print()
        print("Valid Data: ", self.valid_set["data"].shape)
        print("Valid Labels: ", self.valid_set["labels"].shape)
        print("Valid Weights: ", self.valid_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.valid_set["weights"][self.valid_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.valid_set["weights"][self.valid_set["labels"] == 0].sum(),
        )

        holdout_df = get_train_set(selected_indices=holdout_indices)

        self.holdout_set = {
            "labels": holdout_df.pop("labels"),
            "weights": holdout_df.pop("weights"),
            "detailed_labels": holdout_df.pop("detailed_labels"),
            "data": holdout_df,
        }

        del holdout_df

        print()
        print("Holdout Data: ", self.holdout_set["data"].shape)
        print("Holdout Labels: ", self.holdout_set["labels"].shape)
        print("Holdout Weights: ", self.holdout_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.holdout_set["weights"][self.holdout_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.holdout_set["weights"][self.holdout_set["labels"] == 0].sum(),
        )
        print(" \n ")

        print("Training Data: ", self.training_set["data"].shape)
        print(f"DEBUG: model_type = {repr(model_type)}")

        if model_type == "BDT":
            from boosted_decision_tree import BoostedDecisionTree

            self.model = BoostedDecisionTree(name="main")
        elif model_type == "NN":
            from neural_network import NeuralNetwork

            self.nn_params = {
                "layers": [1000, 1000, 1000, 1000, 1000, 1000],
                "dropout": [0.3, 0.3],
                "epochs": 30,
                "batch_size": 32,
            }
            self.model = NeuralNetwork(name="main", **self.nn_params)
        elif model_type == "sample_model":
            from sample_model import SampleModel

            self.model = SampleModel()
        else:
            print(f"model_type {model_type} not found")
            raise ValueError(f"model_type {model_type} not found")
        self.name = model_type

        print(f" Model is { self.name}")

    def fit(self):
        mlflow.set_experiment("NN_experiments")
        # run_name = f"NN_epochs{self.nn_params['epochs']}_bs{self.nn_params['batch_size']} - withSys - with all features"
        run_name = f"NN - all featurse"
        with mlflow.start_run(run_name=run_name):
            # Log model type
            mlflow.log_param("model_type", self.name)
            mlflow.log_param("train_samples", len(self.training_set["data"]))

            if self.name == "NN":
                for k, v in self.nn_params.items():
                    mlflow.log_param(k, v)

            # we balance classes here
            # Balance classes
            balanced_set = self.training_set.copy()
            weights_train = self.training_set["weights"].copy()
            train_labels = self.training_set["labels"].copy()
            class_weights_train = (
                weights_train[train_labels == 0].sum(),
                weights_train[train_labels == 1].sum(),
            )

            for i in range(len(class_weights_train)):
                weights_train[train_labels == i] *= (
                    max(class_weights_train) / class_weights_train[i]
                )

            balanced_set["weights"] = weights_train

            # fitting out model (eg , BDT or NN) with the balanced data
            # Train model
            self.model.fit(
                balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
            )

            # Apply systematics

            # Save info
            self.saved_info = calculate_saved_info(self.model, self.holdout_set)

            # --- Scores
            train_score = self.model.predict(self.training_set["data"])
            holdout_score = self.model.predict(self.holdout_set["data"])
            valid_score = self.model.predict(self.valid_set["data"])
            from sklearn.metrics import accuracy_score

            # After training, calculate accuracy manually on the full training set
            train_preds = (self.model.predict(self.training_set["data"]) > 0.5).astype(
                int
            )
            train_labels = self.training_set["labels"]

            train_acc = accuracy_score(train_labels, train_preds)
            print(f"Final Train Accuracy: {train_acc:.4f}")
            mlflow.log_metric("final_train_accuracy", train_acc)

            valid_preds = (self.model.predict(self.valid_set["data"]) > 0.5).astype(int)
            valid_labels = self.valid_set["labels"]
            valid_acc = accuracy_score(valid_labels, valid_preds)
            mlflow.log_metric("final_valid_accuracy", valid_acc)

            holdout_preds = (self.model.predict(self.holdout_set["data"]) > 0.5).astype(
                int
            )
            holdout_labels = self.holdout_set["labels"]
            holdout_acc = accuracy_score(holdout_labels, holdout_preds)
            mlflow.log_metric("final_holdout_accuracy", holdout_acc)

            # --- mu results
            train_results = compute_mu(
                train_score, self.training_set["weights"], self.saved_info
            )
            holdout_results = compute_mu(
                holdout_score, self.holdout_set["weights"], self.saved_info
            )
            valid_results = compute_mu(
                valid_score, self.valid_set["weights"], self.saved_info
            )

            # --- Print + Log Metrics
            print("Train Results:")
            for key, value in train_results.items():
                print("\t", key, " : ", value)
                mlflow.log_metric(f"train_{key}", value)

            print("Holdout Results:")
            for key, value in holdout_results.items():
                print("\t", key, " : ", value)
                mlflow.log_metric(f"holdout_{key}", value)

            print("Valid Results:")
            for key, value in valid_results.items():
                print("\t", key, " : ", value)
                mlflow.log_metric(f"valid_{key}", value)

            # --- Save score column
            self.valid_set["data"]["score"] = valid_score

            # --- Plots
            from utils import (
                roc_curve_wrapper,
                histogram_dataset,
                stacked_histogram,
                plot_calibration_curve,
            )

            # Create a run folder
            run_dir = "mlruns_temp"
            os.makedirs(run_dir, exist_ok=True)

            """
            Plots per-class score distributions (e.g. score of class 0 vs class 1) for the chosen feature(s), like "score".
            Helps check how well your model separates classes.
            """
            # Histogram
            hist_path = histogram_dataset(
                self.valid_set["data"],
                self.valid_set["labels"],
                self.valid_set["weights"],
                columns=["score"],
                save_path=f"{run_dir}/main_histogram.png",
            )
            mlflow.log_artifact(hist_path)

            # Stacked histogram
            stacked_path = stacked_histogram(
                self.valid_set["data"],
                self.valid_set["labels"],
                self.valid_set["weights"],
                self.valid_set["detailed_labels"],
                "score",
                save_path=f"{run_dir}/main_stacked_histogram.png",
            )
            mlflow.log_artifact(stacked_path)

            # ROC Curve
            roc_path = roc_curve_wrapper(
                score=valid_score,
                labels=self.valid_set["labels"],
                weights=self.valid_set["weights"],
                plot_label="valid_set_" + self.name,
                save_path=f"{run_dir}/main_roc_curve.png",
            )
            mlflow.log_artifact(roc_path)
            self.valid_set["data"]["score"] = valid_score
            self.training_set["data"]["score"] = train_score
            self.holdout_set["data"]["score"] = holdout_score

            labels_train = self.training_set["labels"].values
            print("Unique labels:", np.unique(labels_train))

            # Count how many 0s and 1s
            print("Label counts:", np.bincount(labels_train.astype(int)))

            # Check some features stats per label
            data_train = self.training_set["data"]
            df = pd.DataFrame(data_train)
            df["label"] = labels_train

            print(df.groupby("label").describe())

            calib_path = plot_calibration_curve(
                data_den=self.training_set["data"]["score"].values[
                    self.training_set["labels"].values == 0
                ],
                weight_den=self.training_set["weights"].values[
                    self.training_set["labels"].values == 0
                ],
                data_num=self.training_set["data"]["score"].values[
                    self.training_set["labels"].values == 1
                ],
                weight_num=self.training_set["weights"].values[
                    self.training_set["labels"].values == 1
                ],
                data_denH=self.holdout_set["data"]["score"][
                    self.holdout_set["labels"] == 0
                ],
                weight_denH=self.holdout_set["weights"][
                    self.holdout_set["labels"] == 0
                ],
                data_numH=self.holdout_set["data"]["score"][
                    self.holdout_set["labels"] == 1
                ],
                weight_numH=self.holdout_set["weights"][
                    self.holdout_set["labels"] == 1
                ],
                epsilon=1.0e-20,
                label="Calibration Curve",
                score_range="standard",
                save=f"{run_dir}/main_calibration_curve.pdf",
            )

            mlflow.log_artifact(calib_path)
            from IPython.display import Image, display

            display(Image(filename="nn_architecture.png"))
            mlflow.log_artifact("nn_architecture.png")

            if self.name == "NN" and hasattr(self.model, "history"):
                history = self.model.history
                plt.figure(figsize=(10, 6))
                plt.plot(history["loss"], label="Train Loss")
                plt.plot(history["val_loss"], label="Validation Loss")
                plt.title("Loss Curve")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{run_dir}/main_loss_curve.png", bbox_inches="tight")
                mlflow.log_artifact(f"{run_dir}/main_loss_curve.png")
                plt.close()
            # --- Save trained model
            if self.name == "NN":
                mlflow.keras.log_model(self.model.model, "keras_model")

            self.syst_model = {}

            systs = ["tes"]

            for syst in systs:
                logger.info("Training syst model for %s", syst)
                self.syst_model[syst] = SystModel(
                    NP=syst,
                    systematics=self.systematics,
                    preselection=self.preselection,
                )
                self.syst_model[syst].systematics_values = [1.1, 0.9]
                if not self.istrained:
                    self.syst_model[syst].fit(
                        holdout_set=self.holdout_set, training_set=self.training_set
                    )
                    self.syst_model[syst].save()
                else:
                    try:
                        self.syst_model[syst].load()
                    except Exception as e:
                        logger.error("Error loading syst model: %s", e)
                        self.syst_model[syst].fit(
                            holdout_set=self.holdout_set, training_set=self.training_set
                        )
                        self.syst_model[syst].save()

    def predict(self, test_set):
        """
        Params:
            test_set

        Functionality:
            this function can be used for predictions using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        test_data = test_set["data"]
        test_weights = test_set["weights"]

        if "score" in test_data:
            test_data.pop("score")
        predictions = self.model.predict(test_data)

        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)

        print("Test Results: ", result_mu_cal)

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result
