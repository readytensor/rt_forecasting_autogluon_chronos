import os
import warnings
import joblib
import numpy as np
import pandas as pd
from schema.data_schema import ForecastingSchema
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.exceptions import NotFittedError
from torch import cuda
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from logger import get_logger
from torch import cuda


warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_FILE_NAME = "model.joblib"

logger = get_logger(task_name="model")


class Forecaster:
    """A wrapper class for the Chronos Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    """
    Does Chronos work with covariates or features?
        The current iteration of Chronos does not support covariates or features, 
        however this functionality will be provided in later versions.
        In the meanwhile, presets such as chronos_ensemble combine Chronos with models that do take advantage of features.
    """

    model_name = "Chronos Forecaster - AutoGluon"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        model_alias: str = "base",
        num_samples: int = 30,
        batch_size: int = 16,
        context_length: int = 512,
        optimization_strategy: str = None,
        use_static_features: bool = False,    # static_covariates
        use_future_covariates: bool = False,  # called known_covariates in AutoGluon
        use_past_covariates: bool = False,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new Chronos Forecaster

        Args:

            data_schema (ForecastingSchema):
                The schema of the data.

            model_alias (str):
                The alias of the model to use. The alias is used to load the pre-trained model.
                The alias can be one of "tiny", "mini" , "small", "base", and "large".

            num_samples (int):
                Number of samples used during inference.

            batch_size (int):
                Size of batches used during inference.

            context_length (int): maximum 512
                The context length to use in the model. Shorter context lengths will decrease model accuracy, but result
                in faster inference. If None, the model will infer context length from the data set length at inference time

            optimization_strategy (str):
                Optimization strategy to use for inference on CPUs. If None, the model will use the default implementation.
                If `onnx`, the model will be converted to ONNX and the inference will be performed using ONNX. If ``openvino``,
                inference will be performed with the model compiled to OpenVINO.

            use_static_features (bool):
                Whether the model should use static features if available.

            use_future_covariates (bool):
                Whether the model should use future covariates if available.

            use_past_covariates (bool):
                Whether the model should use past covariates if available.

            random_state (int):
                Sets the underlying random seed at model initialization time.

            **kwargs:
                Optional arguments.
        """
        self.data_schema = data_schema
        self.model_alias = model_alias
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.context_length = context_length
        self.optimization_strategy = optimization_strategy

        self.use_static_features = use_static_features and (
            len(data_schema.static_covariates) > 0
        )
        self.use_past_covariates = (
            use_past_covariates and len(data_schema.past_covariates) > 0
        )
        self.use_future_covariates = use_future_covariates and (
            len(data_schema.future_covariates) > 0
            or self.data_schema.time_col_dtype in ["DATE", "DATETIME"]
        )

        self.random_state = random_state
        self.kwargs = kwargs
        self._is_trained = False
        self.model = None  # set in fit()

    def _prepare_data(self, data: pd.DataFrame) -> TimeSeriesDataFrame:
        """Prepare the data for training or prediction.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            TimeSeriesDataFrame: The prepared data.
        """

        if not self.use_past_covariates and set(self.data_schema.past_covariates).issubset(data.columns):
            data = data.drop(columns=self.data_schema.past_covariates)

        if not self.use_future_covariates and set(self.data_schema.future_covariates).issubset(data.columns):
            data = data.drop(columns=self.data_schema.future_covariates)

        static_features_df = None
        if self.use_static_features:
            static_features_df = data[[
                self.data_schema.id_col]+self.data_schema.static_covariates]
            static_features_df.drop_duplicates(inplace=True, ignore_index=True)

        data = data.drop(columns=self.data_schema.static_covariates)

        prepared_data = TimeSeriesDataFrame.from_data_frame(df=data,
                                                            id_column=self.data_schema.id_col,
                                                            timestamp_column=self.data_schema.time_col,
                                                            static_features_df=static_features_df,
                                                            )

        return prepared_data

    def fit(
        self,
        train_data: pd.DataFrame,
        model_dir_path: str
    ) -> None:
        """Fit the Forecaster model.
        A Chronos model is a pre-trained model, Fitting is mainly to set the model.

        Args:
            train_data (pd.DataFrame): Training data.

        """

        """
        more Hyperparameters:
            device : str, default = None
                Device to use for inference. If None, model will use the GPU if available. For larger model sizes
                `small`, `base`, and `large`; inference will fail if no GPU is available.

            torch_dtype : torch.dtype or {"auto", "bfloat16", "float32", "float64"}, default = "auto"
                Torch data type for model weights, provided to ``from_pretrained`` method of Hugging Face AutoModels. If
                original Chronos models are specified and the model size is ``small``, ``base``, or ``large``, the
                ``torch_dtype`` will be set to ``bfloat16`` to enable inference on GPUs.

            data_loader_num_workers : int, default = 0
                Number of worker processes to be used in the data loader. See documentation on ``torch.utils.data.DataLoader``
                for more information.
        """
        prepared_data = self._prepare_data(train_data)

        future_covariates = None
        if self.use_future_covariates:
            future_covariates = self.data_schema.future_covariates

        self.model = TimeSeriesPredictor(path=os.path.join(model_dir_path, MODEL_FILE_NAME),
                                         target=self.data_schema.target,
                                         prediction_length=self.data_schema.forecast_length,
                                         known_covariates_names=future_covariates,
                                         cache_predictions=False,
                                         ).fit(
            train_data=prepared_data,
            hyperparameters={
                "Chronos": {
                    "model_path": self.model_alias,
                    "batch_size": self.batch_size,
                    "num_samples": self.num_samples,
                    "context_length": self.context_length,
                    "optimization_strategy": self.optimization_strategy,
                    "random_seed": self.random_state,
                }
            },
            skip_model_selection=True,
            verbosity=0,
        )
        self._is_trained = True

    def predict(
        self, train_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.
        Args:
            train_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The predictions dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        prepared_data = self._prepare_data(train_data)
        predictions = self.model.predict(data=prepared_data, use_cache=False)
        predictions.reset_index(inplace=True)

        predictions = predictions.rename(columns={"item_id": self.data_schema.id_col,
                                                  "timestamp": self.data_schema.time_col,
                                                  "mean": prediction_col_name})

        if self.data_schema.time_col_dtype in ["INT", "OTHER"]:
            last_timestamp = train_data[self.data_schema.time_col].max()
            new_timestamps = np.arange(
                last_timestamp + 1, last_timestamp + 1 + self.data_schema.forecast_length
            )
            predictions[self.data_schema.time_col] = np.tile(
                new_timestamps, predictions[self.data_schema.id_col].nunique())

        return predictions[[self.data_schema.id_col, self.data_schema.time_col, prediction_col_name]]

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        self.model.save()
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @ classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        forecaster = joblib.load(os.path.join(
            model_dir_path, PREDICTOR_FILE_NAME))
        model = TimeSeriesPredictor.load(
            os.path.join(model_dir_path, MODEL_FILE_NAME))
        forecaster.model = model
        return forecaster

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    data_schema: ForecastingSchema,
    train_data: pd.DataFrame,
    model_dir_path: str,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(train_data=train_data, model_dir_path=model_dir_path)
    return model


def predict_with_model(
    model: Forecaster, train_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        train_data (pd.DataFrame): The train input data for forecasting used to do prediction.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(train_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)
