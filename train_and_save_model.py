import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    df = pd.read_csv('refined_remittance_outflows.csv')
    df['Hist_Median'] = df.groupby('Country')['Remittance_Outflow_USD_Mil'].transform('median')
    df['Year'] = df['Year'].astype(int)

    X = df[['Year', 'Hist_Median']]
    y = df['Remittance_Outflow_USD_Mil']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f'Trained RandomForestRegressor: RMSE={rmse:.2f}, R2={r2:.4f}')

    with open('remittance_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('Saved trained model to remittance_model.pkl')


if __name__ == '__main__':
    main()
