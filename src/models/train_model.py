import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler


def train_and_evaluate_model(train_data, test_data, predictors, target):
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[predictors])
    X_test = scaler.transform(test_data[predictors])

    # Try ridge regression with hyperparameter tuning
    ridge = Ridge()
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, train_data[target])

    best_alpha = grid_search.best_params_['alpha']
    print(f"Best alpha: {best_alpha}")

    # Use best model
    reg = Ridge(alpha=best_alpha)
    reg.fit(X_train, train_data[target])

    # Also try a Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, train_data[target])

    # Make predictions with both models
    train_predictions_ridge = reg.predict(X_train)
    test_predictions_ridge = reg.predict(X_test)

    train_predictions_rf = rf.predict(X_train)
    test_predictions_rf = rf.predict(X_test)

    # Evaluate models
    print("\nRidge Regression Performance:")
    print(f"Train R² Score: {r2_score(train_data[target], train_predictions_ridge):.3f}")
    print(f"Test R² Score: {r2_score(test_data[target], test_predictions_ridge):.3f}")
    print(f"Train MAE: {mean_absolute_error(train_data[target], train_predictions_ridge):.3f}")
    print(f"Test MAE: {mean_absolute_error(test_data[target], test_predictions_ridge):.3f}")

    print("\nRandom Forest Performance:")
    print(f"Train R² Score: {r2_score(train_data[target], train_predictions_rf):.3f}")
    print(f"Test R² Score: {r2_score(test_data[target], test_predictions_rf):.3f}")
    print(f"Train MAE: {mean_absolute_error(train_data[target], train_predictions_rf):.3f}")
    print(f"Test MAE: {mean_absolute_error(test_data[target], test_predictions_rf):.3f}")

    # Feature importance for ridge
    feature_importance_ridge = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': reg.coef_
    })

    # Feature importance for RF
    feature_importance_rf = pd.DataFrame({
        'Feature': predictors,
        'Importance': rf.feature_importances_
    })

    print("\nRidge Feature Importance:")
    print(feature_importance_ridge.sort_values('Coefficient', ascending=False).head(10))

    print("\nRandom Forest Feature Importance:")
    print(feature_importance_rf.sort_values('Importance', ascending=False).head(10))

    # Cross-validation score
    cv_scores = cross_val_score(reg, X_train, train_data[target], cv=5)
    print(f"\nCross-validation R² scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Plot actual vs predicted for the better model
    better_model = "Ridge" if r2_score(test_data[target], test_predictions_ridge) > r2_score(test_data[target],
                                                                                             test_predictions_rf) else "RF"

    if better_model == "Ridge":
        train_predictions = train_predictions_ridge
        test_predictions = test_predictions_ridge
        chosen_model = reg
    else:
        train_predictions = train_predictions_rf
        test_predictions = test_predictions_rf
        chosen_model = rf

    plt.figure(figsize=(10, 6))
    plt.scatter(train_data[target], train_predictions, alpha=0.5, label='Train')
    plt.scatter(test_data[target], test_predictions, alpha=0.5, label='Test')
    plt.plot([0, max(train_data[target])], [0, max(train_data[target])], 'r--')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.title(f'Actual vs Predicted {target} using {better_model}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return chosen_model, scaler, feature_importance_ridge if better_model == "Ridge" else feature_importance_rf
