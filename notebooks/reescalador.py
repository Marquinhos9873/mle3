import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA


class scaler_process:
    def __init__(self, data: pd.DataFrame, scale_method, pipeline_name: str):
        self.scale_method = scale_method
        self.data = data
        self.pca = PCA(n_components=2)
        self.pipeline_name = pipeline_name
        self.explained_variance = None

    def run_scaler(self):
        scaler_pipeline = Pipeline(
            steps=[
                ("scaler", self.scale_method),
                ("pca", self.pca),
            ]
        )

        transformed = scaler_pipeline.fit_transform(self.data)
        self.explained_variance = self.pca.explained_variance_ratio_
        metadata = pd.DataFrame(
            transformed,
            columns=[f"{self.pipeline_name}_feat_{i+1}" for i in range(self.pca.n_components_)]
        )

        return scaler_pipeline, metadata

#### sacler_table = scaler(data = no_gender_data, scale_method = StandarScaler(); MacxMinScaler(), RobustScaler(), pipeli)

    def plot_variance(self):
        if self.explained_variance is None:
            raise ValueError("Ejecuta run_scaler() antes de graficar la varianza.")

        plt.plot(range(1, len(self.explained_variance) + 1), self.explained_variance, marker="o")
        plt.title("Varianza Explicada por Componente")
        plt.xlabel("Componente")
        plt.ylabel("Proporción de Varianza Explicada")
        plt.show()
### scaler_table.plot_variance()