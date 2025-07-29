import numpy as np
from sys import path
import pickle
import matplotlib.pyplot as plt
from iminuit import Minuit
from checks import check_reweighting, check_calibration, roc_curve
import os
import pandas as pd
from iminuit import Minuit
import logging
from utils import plot_calibration_curve, roc_curve_wrapper, plot_score_distributions
import mlflow


path.append("../")
path.append("../ingestion_program")


from checks import plot_NLL

log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

# XGBOOST = True
XGBOOST = False

# TENSORFLOW = not XGBOOST
TENSORFLOW = True


current_dir = os.path.dirname(__file__)


class SystModel:
    """
    This class implements a model for the SBI submission.
    """

    def __init__(self, NP="tes", systematics=None, preselection=None):

        self.systematics = systematics
        self.systematics_values = [1.1, 0.9]
        self.re_train = True
        self.norm_syst = False
        self.lower_bound = 0.01
        self.upper_bound = 0.99
        self.NP = NP
        self.preselection = preselection

        if XGBOOST == True:
            from boosted_decision_tree import BoostedDecisionTree

            self.models = {
                "plus": BoostedDecisionTree(name=f"{self.NP}_plus"),
                "minus": BoostedDecisionTree(name=f"{self.NP}_minus"),
            }
            self.name = "BDT"

            self.models["plus"].calibrate = True
            self.models["minus"].calibrate = True

        elif TENSORFLOW == True:
            from neural_network import NeuralNetwork

            self.models = {
                "plus": NeuralNetwork(name=f"{self.NP}_plus"),
                "minus": NeuralNetwork(name=f"{self.NP}_minus"),
            }
            self.name = "NN"

        else:
            logger.error("No model selected")
            raise ValueError("No model selected")

        self.categories = ["ztautau", "ttbar", "diboson"]

        self.plots_dir = os.path.join(current_dir, "plots")
        self.model_dir = os.path.join(current_dir, "models")

        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # for key in self.models.keys():
        #     istrained = self.models[key].load(self.model_dir)
        self.istrained = False

    def density_ratio(self, score):
        logger.debug("score shape %s", score.shape)

        # Create a copy of the score array to avoid modifying the original
        score_new = score.copy()

        # Replace values greater than self.upper_bound with 0.99
        score_new[score > self.upper_bound] = self.upper_bound
        score_new[score < self.lower_bound] = self.lower_bound

        logger.debug("score_new shape %s", score_new.shape)

        return (1 - score_new) / (score_new)

    def train(self, dataset):

        self.model.fit(dataset["data"], dataset["labels"], dataset["weights"])

    def predict_model(self, data):

        scores = {}
        density_ratios = {}
        for key in self.models.keys():

            model = self.models[key]
            score = model.predict(data["data"][self.columns])

            scores[key] = score
            density_ratios[key] = self.density_ratio(score)

        return scores, density_ratios

    def analyze(
        self,
        score,
        data,
        category="",
        calib_label="",
        training_set=None,
        holdout_set=None,
    ):

        check_calibration(
            score,
            data["labels"],
            data["weights"],
            bins=50,
            filename=self.plots_dir + f"/{category}_calibration_pre2.png",
            label=calib_label,
        )
        run_dir = "mlruns_temp"
        os.makedirs(run_dir, exist_ok=True)

        calib_path = plot_calibration_curve(
            data_den=training_set["data"]["score"].values[
                training_set["labels"].values == 0
            ],
            weight_den=training_set["weights"].values[
                training_set["labels"].values == 0
            ],
            data_num=training_set["data"]["score"].values[
                training_set["labels"].values == 1
            ],
            weight_num=training_set["weights"].values[
                training_set["labels"].values == 1
            ],
            data_denH=holdout_set["data"]["score"].values[
                holdout_set["labels"].values == 0
            ],
            weight_denH=holdout_set["weights"][holdout_set["labels"].values == 0],
            data_numH=holdout_set["data"]["score"].values[
                holdout_set["labels"].values == 1
            ],
            weight_numH=holdout_set["weights"][holdout_set["labels"].values == 1],
            epsilon=1.0e-20,
            label="Calibration Curve",
            score_range="standard",
            save=f"{run_dir}/calibration_curve_{category}.pdf",
        )
        mlflow.log_artifact(calib_path)
        print(calib_path)
        density_ratio = self.density_ratio(score)

        # ROC curve
        roc_path = roc_curve_wrapper(
            score=score,
            labels=data["labels"],
            weights=data["weights"],
            plot_label=f"Holdout_{category}",
            save_path=f"{run_dir}/roc_curve_{category}.png",
        )
        print(roc_path)
        mlflow.log_artifact(roc_path)

        # data["data"][f"score_{category}"] = score

        reweighting_paths = check_reweighting(density_ratio, data, category)
        for path in reweighting_paths:
            print(path)
            mlflow.log_artifact(path)
        print("score distribution graphs")

        score_dist_path = plot_score_distributions(
            training_set,
            holdout_set,
            self.preselection,
            self.columns,
            self.models,
            self.plots_dir,
            self.NP,
        )
        mlflow.log_artifact(score_dist_path)

    def fit(self, holdout_set, training_set=None):
        """
        Trains the model.

        Params:
            None

        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.

        Returns:
            None
        """
        holdout_data_sets = self.systematics_datasets(holdout_set)

        self.Gx = {}

        for key in holdout_data_sets.keys():

            holdout_data_set = self.preselection.apply_pre_selection(
                holdout_data_sets[key],
                threshold=0.8,
            )

            self.Gx[key] = np.sum(
                holdout_data_set["weights"][holdout_data_set["labels"] == 1]
            ) / np.sum(holdout_data_set["weights"][holdout_data_set["labels"] == 0])

            logger.debug("Gx %s", self.Gx[key])
            logger.debug("Gx shape %s", self.Gx[key].shape)

        if self.norm_syst:
            return

        if not self.istrained:

            if training_set is None:
                logger.error("Training set not provided")
                raise ValueError("Training set not provided")

            training_data_sets = self.systematics_datasets(training_set)

            for key in training_data_sets.keys():
                balanced_set = training_data_sets[key].copy()
                weights_train = training_data_sets[key]["weights"].copy()
                train_labels = training_data_sets[key]["labels"].copy()
                class_weights_train = (
                    weights_train[train_labels == 0].sum(),
                    weights_train[train_labels == 1].sum(),
                )

                for i in range(len(class_weights_train)):
                    weights_train[train_labels == i] *= (
                        max(class_weights_train) / class_weights_train[i]
                    )

                balanced_set["weights"] = weights_train

                print(balanced_set["data"].columns)

                print(len(weights_train[train_labels == 1]), f"label = 1 {key}")
                print(len(weights_train[train_labels == 0]), f"label = 0 {key}")

                self.models[key].fit(
                    balanced_set["data"],
                    balanced_set["labels"],
                    balanced_set["weights"],
                )

                # self.models[key].save(self.model_dir)
                del balanced_set

        # read and clean holdout set

        for key in holdout_data_sets.keys():

            holdout_score_temp = self.models[key].predict(
                holdout_data_sets[key]["data"],
            )
            calib_label = "Calibrated" if self.models[key].calibrate else "Uncalibrated"
            holdout_data_sets[key]["data"]["score"] = holdout_score_temp
            training_data_sets[key]["data"]["score"] = self.models[key].predict(
                training_data_sets[key]["data"]
            )
            self.analyze(
                score=holdout_score_temp,
                data=holdout_data_sets[key],
                category=key,
                calib_label=calib_label,
                training_set=training_data_sets[key],
                holdout_set=holdout_data_sets[key],
            )
            # test
            print(
                training_data_sets[key]["data"].shape,
                training_data_sets[key]["data"].columns,
            )
            print(
                holdout_data_sets[key]["data"].shape,
                holdout_data_sets[key]["data"].columns,
            )

        holdout_set_norm = self.systematics(holdout_set)

        preselected_holdout = self.preselection.apply_pre_selection(
            holdout_set_norm,
            threshold=0.8,
        )

        _, holdout_density_ratios = self.predict_model(preselected_holdout)

        self.fit_extropolate(holdout_density_ratios)

        Gx_0_holdout = self.syst_fun(self.coeff_G, 1)
        gx_0_holdout = self.syst_fun(self.coeff_g, 1)

        Gx_plus = self.Gx["plus"]
        Gx_minus = self.Gx["minus"]

        logger.debug("Gx_plus %s", Gx_plus)
        logger.debug("Gx_minus %s", Gx_minus)

        gx_plus = self.predict(preselected_holdout)["plus"]
        gx_minus = self.predict(preselected_holdout)["minus"]

        logger.debug("Gx_0_holdout %s", Gx_0_holdout)
        logger.debug("gx_0_holdout %s", gx_0_holdout)

        pdf_holdout = self.pdf(1, gx_plus, gx_minus)
        pdf_holdout_plus = self.pdf(1.01, gx_plus, gx_minus)
        pdf_holdout_minus = self.pdf(0.99, gx_plus, gx_minus)

        logger.debug("pdf_holdout %s", pdf_holdout)
        logger.debug("pdf_holdout_plus %s", pdf_holdout_plus)
        logger.debug("pdf_holdout_minus %s", pdf_holdout_minus)

    def syst_fun(self, coeff, alpha):
        return 1 + coeff[0] * alpha + coeff[1] * alpha**2

    def fit_extropolate(self, density_ratios):

        # creates a 3d polynomial fit for the density ratios, which depend on alpha
        # the fit is used to extrapolate the density ratios to alpha = 1 + 0.01 and alpha = 1 - 0.01

        def cost_g(coeff_0, coeff_1):
            coeff = [coeff_0, coeff_1]
            return np.sum(
                (
                    self.syst_fun(coeff, self.systematics_values[0])
                    - density_ratios["plus"]
                )
                ** 2
                + (
                    self.syst_fun(coeff, self.systematics_values[1])
                    - density_ratios["minus"]
                )
                ** 2
            )

        m = Minuit(cost_g, coeff_0=0, coeff_1=0)
        m.migrad()

        self.coeff_g = [m.values["coeff_0"], m.values["coeff_1"]]

        def cost_G(coeff_0, coeff_1):
            coeff = [coeff_0, coeff_1]
            return np.sum(
                (self.syst_fun(coeff, self.systematics_values[0]) - self.Gx["plus"])
                ** 2
                + (self.syst_fun(coeff, self.systematics_values[1]) - self.Gx["minus"])
                ** 2
            )

        m = Minuit(cost_G, coeff_0=0, coeff_1=0)
        m.migrad()

        self.coeff_G = [m.values["coeff_0"], m.values["coeff_1"]]

    def pdf(self, alpha, g_x_plus, g_x_minus):

        if alpha > self.systematics_values[0]:
            pdf = (self.Gx["plus"] * g_x_plus) ** alpha
        elif alpha < self.systematics_values[1]:
            pdf = (self.Gx["minus"] * g_x_minus) ** (-alpha)
        else:
            pdf = (
                self.syst_fun(self.coeff_g, alpha)
                * self.syst_fun(self.coeff_G, alpha)
                * np.ones(len(g_x_plus))
            )

        logger.debug("pdf shape %s", pdf.shape)

        return pdf

    def predict(self, data):
        """
        Predicts the class of the data.

        Args:
            data (dict): A dictionary containing the data.

        Returns:
            dict: A dictionary containing the predicted class of the data.
        """
        data.pop("score", None)
        # data = data[self.columns]
        data["data"] = data["data"][self.columns]
        _, density_ratios = self.predict_model(data)

        return density_ratios

    def systematics_datasets(self, dataset):
        """
        Add systematics to the dataset.

        Args:
            dataset (dict): A dictionary containing the dataset.

        Returns:
            dict: A dictionary containing the dataset with systematics.
        """

        # self.columns = [
        #     "PRI_had_pt",
        #     "PRI_met",
        #     "PRI_met_phi",
        #     "DER_mass_transverse_met_lep",
        #     "DER_mass_vis",
        #     "DER_pt_h",
        #     "DER_deltar_had_lep",
        #     "DER_pt_tot",
        #     "DER_sum_pt",
        #     "DER_pt_ratio_lep_had",
        #     "DER_met_phi_centrality",
        # ]

        self.columns = [
            "PRI_lep_pt",
            "PRI_lep_eta",
            "PRI_lep_phi",
            "PRI_had_pt",
            "PRI_had_eta",
            "PRI_had_phi",
            "PRI_jet_leading_pt",
            "PRI_jet_leading_eta",
            "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt",
            "PRI_jet_subleading_eta",
            "PRI_jet_subleading_phi",
            "PRI_n_jets",
            "PRI_jet_all_pt",
            "PRI_met",
            "PRI_met_phi",
            "DER_mass_transverse_met_lep",
            "DER_mass_vis",
            "DER_pt_h",
            "DER_deltaeta_jet_jet",
            "DER_mass_jet_jet",
            "DER_prodeta_jet_jet",
            "DER_deltar_had_lep",
            "DER_pt_tot",
            "DER_sum_pt",
            "DER_pt_ratio_lep_had",
            "DER_met_phi_centrality",
            "DER_lep_eta_centrality",
        ]

        syst_fixed_setting = {
            "tes": 1.0,
            "bkg_scale": 1.0,
            "jes": 1.0,
            "soft_met": 0.0,
            "ttbar_scale": 1.0,
            "diboson_scale": 1.0,
        }

        syst_setting = syst_fixed_setting.copy()

        syst_setting[self.NP] = 1.0
        dataset_nom = self.systematics(
            dataset.copy(), dopostprocess=True, **syst_fixed_setting
        )

        df_nom = dataset_nom["data"]
        df_nom["labels"] = np.zeros(len(df_nom))
        df_nom["weights"] = dataset_nom["weights"]  # <-- Add this line

        print(df_nom["weights"].sum(), "111sum of the weights of df nom")

        del dataset_nom

        data_sets = {}

        if self.NP in syst_setting.keys():

            for syst_value in self.systematics_values:
                syst_setting[self.NP] = syst_value

                name = "minus" if syst_value < syst_fixed_setting[self.NP] else "plus"
                print("syst_value", syst_value)
                print(self.NP, " :Hello tis is NP")
                print("this is the shifted for the ", name)
                print(syst_fixed_setting, "syst_fixed_setting")
                print(syst_setting, "syst_setting")

                dataset_syst = self.systematics(
                    dataset.copy(), dopostprocess=True, **syst_setting
                )
                df_syst = dataset_syst["data"]

                df_syst["labels"] = np.ones(len(df_syst))

                df_syst["weights"] = dataset_syst["weights"]  # <-- Add this line

                print(df_syst["weights"].sum(), "sum of the weights of df sys")
                print(df_nom["weights"].sum(), "sum of the weights of df nom")

                print(df_syst["weights"].count(), "sum of the weights of df sys")
                print(df_nom["weights"].count(), "sum of the weights of df nom")

                df = pd.concat([df_nom, df_syst])

                df = df.sample(frac=1)

                labels = df.pop("labels")
                df.pop("score")
                weights = df.pop("weights")

                df = df[self.columns]

                """
                {
                "labels": pd.Series,         # shape: (5000,)
                "weights": pd.Series,        # shape: (5000,)
                "detailed_labels": pd.Series, # shape: (5000,)
                "data": pd.DataFrame         # shape: (5000, 28)
                }
                """

                data_sets[f"{name}"] = {
                    "data": df,
                    "labels": labels,
                    "weights": weights,
                }
                del df

        else:
            raise ValueError("Systematics not implemented")

        return data_sets

    def save(self):

        # saving coefficients
        with open(self.model_dir + "/coeff_g.pkl", "wb") as f:
            pickle.dump(self.coeff_g, f)

        with open(self.model_dir + "/coeff_G.pkl", "wb") as f:
            pickle.dump(self.coeff_G, f)

        # saving Gx
        with open(self.model_dir + "/Gx.pkl", "wb") as f:
            pickle.dump(self.Gx, f)

        logger.info("Model saved")

    def load(self):

        with open(self.model_dir + "/coeff_g.pkl", "rb") as f:
            self.coeff_g = pickle.load(f)

        with open(self.model_dir + "/coeff_G.pkl", "rb") as f:
            self.coeff_G = pickle.load(f)

        with open(self.model_dir + "/Gx.pkl", "rb") as f:
            self.Gx = pickle.load(f)

        logger.info("Model loaded")
