import pandas as pd
import lightgbm as lgb
from train import train_lightgbm
from sklearn.model_selection import train_test_split
from titanic_survival_prediction import TitanicSurvivalPredictionService

if __name__ == '__main__':
    # TODO : Data Loading Part
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    train_df.head()

    y = train_df.pop('Survived')
    cols = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']
    X_train, X_test, y_train, y_test = train_test_split(train_df[cols],
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Split Data
    train_data = lgb.Dataset(data=X_train[cols],
                             label=y_train)

    test_data = lgb.Dataset(data=X_test[cols],
                            label=y_test)

    model = train_lightgbm(train_data, test_data)

    # Save Bento Service
    bento_service = TitanicSurvivalPredictionService()
    bento_service.pack('model', model)

    saved_path = bento_service.save()

    # Model Validation Part
    # TODO : Create Model Validator
    change_model = True

    if change_model:
        # TODO : Model Deploy (CI/CD에서 처리하는게 더 좋을듯) => 그렇다면 DB 하나 띄우고 저장한 후,  모델 성능 비교. 룰도 지정 필요. 데이터셋 지정도 고민 필요
        pass
    else:
        pass
