import bentoml
import pandas
from bentoml.frameworks.lightgbm import LightGBMModelArtifact
from bentoml.adapters import DataframeInput


# TODO : ver를 지정할 수 있는데, 버전 관리는 어떻게 할 것인가?
@bentoml.ver(major=1, minor=0)
@bentoml.artifacts([LightGBMModelArtifact('model')])
@bentoml.env(pip_packages=['lightgbm==2.3.1'])
class TitanicSurvivalPredictionService(bentoml.BentoService):

    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        self.validate_data(df)
        data = self.preprocess(df)
        print(f"Prediction! data : {data}")
        # TODO : 이 로그 prediction.log에 쌓이는데 프로메테우스에 넘기기?
        return self.artifacts.model.predict(data)

    def preprocess(self, df):
        # TODO : Preprocess

        data = df[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']]
        return data

    def validate_data(self, df):
        # TODO : Data Validation Condition
        assert type(df) == pandas.core.frame.DataFrame
