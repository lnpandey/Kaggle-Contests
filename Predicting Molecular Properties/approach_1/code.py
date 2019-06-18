from funcs import *
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import time


# # train = pd.read_csv(f'{file_folder}/train.csv')
# test = pd.read_csv(f'{file_folder}/test.csv')
# # # sub = pd.read_csv(f'{file_folder}/sample_submission.csv')
# structures = pd.read_csv(f'{file_folder}/structures.csv')
# # potential_energy = pd.read_csv(f'{file_folder}/potential_energy.csv')
# # dipole_moments = pd.read_csv(f'{file_folder}/dipole_moments.csv')
# # mulliken_charges = pd.read_csv(f'{file_folder}/mulliken_charges.csv')
# # magnetic_shielding_tensors = pd.read_csv(f'{file_folder}/magnetic_shielding_tensors.csv')
# # scalar_coupling_contributions = pd.read_csv(f'{file_folder}/scalar_coupling_contributions.csv')
# #
# # mol_atom = pd.merge(mulliken_charges, magnetic_shielding_tensors, how='inner')
# # mol = pd.merge(potential_energy, dipole_moments, how='inner')
# #
# # print(mol_atom.columns, mol.columns)
# #
# # extra_data = pd.merge(mol_atom, mol, how='inner')
# # data = pd.merge(extra_data, structures, how='inner')
# #
# # train = pd.merge(train, scalar_coupling_contributions, how='inner')
# # train = map_atom_info(train, data, 0)
# # train = map_atom_info(train, data, 1)
# # # train.to_csv('data/full_data.csv', index=False)
#
# train = pd.read_csv(f'{file_folder}/full_data.csv')
#
# print('\n\nImported all!\n\n')
#
#
# test = map_atom_info(test, structures, 0)
# test = map_atom_info(test, structures, 1)
#
# train_p_0 = train[['x_0', 'y_0', 'z_0']].values
# train_p_1 = train[['x_1', 'y_1', 'z_1']].values
# test_p_0 = test[['x_0', 'y_0', 'z_0']].values
# test_p_1 = test[['x_1', 'y_1', 'z_1']].values
#
# train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
# test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
# train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
# test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
# train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
# test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
# train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
# test['dist_z'] = (test['z_0'] - test['z_1']) ** 2
#
# train['type_0'] = train['type'].apply(lambda x: x[0])
# test['type_0'] = test['type'].apply(lambda x: x[0])
#
# print('Step 1')
#
# train = create_features(train)
# test = create_features(test)
# print('Step 2')
#
# for f in ['atom_0', 'atom_1', 'type', 'type_0']:
#     lbl = LabelEncoder()
#     lbl.fit(list(train[f].values) + list(test[f].values))
#     train[f] = lbl.transform(list(train[f].values))
#     test[f] = lbl.transform(list(test[f].values))
# print('Step 3')
#
# X = train[good_columns].copy()
# y = train['scalar_coupling_constant']
# X_test = test[good_columns].copy()
# Y_mid = train[['fc', 'sd', 'pso', 'dso']].copy()
# X_mid = train[['mulliken_charge_x', 'XX_x', 'YX_x', 'ZX_x', 'XY_x', 'YY_x', 'ZY_x', 'XZ_x', 'YZ_x', 'ZZ_x','potential_energy_x',
#      'X_x', 'Y_x', 'Z_x', 'mulliken_charge_y', 'XX_y', 'YX_y', 'ZX_y', 'XY_y', 'YY_y', 'ZY_y', 'XZ_y', 'YZ_y',
#      'ZZ_y', 'potential_energy_y', 'X_y', 'Y_y', 'Z_y']].copy()  # print(train[['atom_index_0', 'atom_0', 'atom_1']])
#
# print('Saving')
# X.to_csv('data/X.csv', index=False)
# y.to_csv('data/y.csv', index=False)
# X_test.to_csv('data/X_test.csv', index=False)
# Y_mid.to_csv('data/Y_mid.csv', index=False)
# X_mid.to_csv('data/X_mid.csv', index=False)
# print('Done')

def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None,
                           plot_feature_importance=False, model=None,
                           verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                                  'catboost_metric_name': 'MAE',
                                  'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1, )

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                      **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    # if model_type == 'lgb':
    #     if plot_feature_importance:
    #         feature_importance["importance"] /= folds.n_splits
    #         cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
    #             by="importance", ascending=False)[:50].index
    #
    #         best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
    #
    #         plt.figure(figsize=(16, 12));
    #         sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
    #         plt.title('LGB Features (avg over folds)');
    #
    #         result_dict['feature_importance'] = feature_importance

    return result_dict


X = pd.read_csv(f'{file_folder}/X.csv')
y = pd.read_csv(f'{file_folder}/y.csv')
# Y_mid = pd.read_csv(f'{file_folder}/Y_mid.csv')
X_test = pd.read_csv(f'{file_folder}/X_test.csv')
# X_mid = pd.read_csv(f'{file_folder}/X_mid.csv')
sub = pd.read_csv(f'{file_folder}/sample_submission.csv')

params = {'num_leaves': 128, 'min_child_samples': 79, 'objective': 'regression', 'max_depth': 9, 'learning_rate': 0.2,
          "boosting_type": "gbdt", "subsample_freq": 1, "subsample": 0.9, "bagging_seed": 11,
          "metric": 'mae', "verbosity": -1, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'colsample_bytree': 1.0
          }
X_short = pd.DataFrame({'ind': list(X.index), 'type': X['type'].values.flatten(), 'oof': [0] * len(X), 'target': y.values.flatten()})
X_short_test = pd.DataFrame({'ind': list(X_test.index), 'type': X_test['type'].values.flatten(), 'prediction': [0] * len(X_test)})
n_fold = 3
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
for t in X['type'].unique():
    print(f'Training of type {t}')
    X_t = X.loc[X['type'] == t]
    X_test_t = X_test.loc[X_test['type'] == t]
    y_t = X_short.loc[X_short['type'] == t, 'target']
    result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds,
                                              model_type='lgb', eval_metric='group_mae', plot_feature_importance=False,
                                              verbose=500, early_stopping_rounds=200, n_estimators=3000)
    X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb3['oof']
    X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict_lgb3['prediction']

sub['scalar_coupling_constant'] = X_short_test['prediction']

print('\n\nImported all!\n\n')

X = X.fillna(X.mean())
# X=(X-X.min())/(X.max()-X.min())
# X_mid=(X_mid-X_mid.min())/(X_mid.max()-X_mid.min())
# X = X.drop(columns=['atom_0'])

sub.to_csv(f'{file_folder}/pred.csv', index=False)
sub.head()
