from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA





class Scaler_process:
    def __init__(self, data: pd.DataFrame, scale_method: "method", pipeline_name: str):
        self.scale_method = scale_method
        self.data = data
        self.pca = PCA(n_components=2)
        self.pipeline_name = pipeline_name
        self.explained_variance = self.pca.explained_variance_ratio_

    def run_scaler():
        
        scaler_pipeline = Pipeline(
            steps = [
                ("scaler", self.scale_method),
                ("PCA", self.pca),
                    ]
            )

        metada = pd.DataFrame(
            scaler_pipeline.fit_transform(self.data),
            columns =  [f"{self.pipeline_name}_feat_{i+1}" for i in range(self.n_components)]
        )
        
        
        return scaler_pipeline, metadata

#### sacler_table = scaler(data = no_gender_data, scale_method = StandarScaler(); MacxMinScaler(), RobustScaler(), pipeli)

    def plot_variance(self):
            plt.plot(range(1, len(self.explained_variance)+1), self.explained_variance, marker="o")
            plt.title("Explained Variance by Components")
            plt.xlabel("Component")
            plt.ylabel("Explained Variance Ratio")
            plt.show()
### scaler_table.plot_variance()