import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def train_and_evaluate_model(train_data, test_data, predictors, target):
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[predictors])
    X_test = scaler.transform(test_data[predictors])

    # Train model
    reg = LinearRegression()
    reg.fit(X_train, train_data[target])

    # Make predictions
    train_predictions = reg.predict(X_train)
    test_predictions = reg.predict(X_test)

    # Evaluate model
    train_r2 = r2_score(train_data[target], train_predictions)
    test_r2 = r2_score(test_data[target], test_predictions)

    # Print results
    print("\nModel Performance:")
    print(f"Train R² Score: {train_r2:.3f}")
    print(f"Test R² Score: {test_r2:.3f}")

    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': reg.coef_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('Coefficient', ascending=False))

    # Cross-validation score
    cv_scores = cross_val_score(reg, X_train, train_data[target], cv=5)
    print(f"\nCross-validation R² scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(train_data[target], train_predictions, alpha=0.5, label='Train')
    plt.scatter(test_data[target], test_predictions, alpha=0.5, label='Test')
    plt.plot([0, max(train_data[target])], [0, max(train_data[target])], 'r--')
    plt.xlabel('Actual Stint Length')
    plt.ylabel('Predicted Stint Length')
    plt.title('Actual vs Predicted Stint Length')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return reg, scaler, feature_importance
