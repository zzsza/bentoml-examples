import lightgbm as lgb


def train_lightgbm(train_data, test_data):
    lgb_params = {
        'boosting': 'dart',  # dart (drop out trees) often performs better
        'application': 'binary',  # Binary classification
        'learning_rate': 0.05,  # Learning rate, controls size of a gradient descent step
        'min_data_in_leaf': 20,  # Data set is quite small so reduce this a bit
        'feature_fraction': 0.7,  # Proportion of features in each boost, controls overfitting
        'num_leaves': 41,  # Controls size of tree since LGBM uses leaf wise splits
        'metric': 'binary_logloss',  # Area under ROC curve as the evaulation metric
        'drop_rate': 0.15
    }

    evaluation_results = {}
    model = lgb.train(train_set=train_data,
                      params=lgb_params,
                      valid_sets=[train_data, test_data],
                      valid_names=['Train', 'Test'],
                      evals_result=evaluation_results,
                      num_boost_round=500,
                      early_stopping_rounds=100,
                      verbose_eval=20
                      )

    return model
