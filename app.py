import pickle
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


@st.cache_data
def load_data():
    df = pd.read_csv('refined_remittance_outflows.csv')
    df['Year'] = df['Year'].astype(int)
    df['Hist_Median'] = df.groupby('Country')['Remittance_Outflow_USD_Mil'].transform('median')
    return df


def train_and_save_model(df):
    model_path = os.path.join(os.path.dirname(__file__), 'remittance_model.pkl')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[['Year', 'Hist_Median']]
    y = df['Remittance_Outflow_USD_Mil']
    model.fit(X, y)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'remittance_model.pkl')
    if not os.path.exists(model_path):
        df = load_data()
        with st.spinner('Training model because remittance_model.pkl is not present...'):
            return train_and_save_model(df)
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_outflow(model, year, hist_median):
    predicted = model.predict([[year, hist_median]])[0]
    return max(predicted, 0.0)


def main():
    st.set_page_config(
        page_title='Remittance Outflow Predictor',
        page_icon='💸',
        layout='wide',
    )

    st.markdown(
        "<div style='padding: 0 20px;'>"
        "<h1 style='color:#1F2937;'>Global Remittance Outflow Forecast</h1>"
        "<p style='font-size:1.05rem; color:#4B5563;'>Use the historical dataset and a trained Random Forest model to forecast country-level remittance outflow in USD millions.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    df = load_data()
    model = load_model()

    countries = sorted(df['Country'].unique())
    default_country = 'United States' if 'United States' in countries else countries[0]

    with st.sidebar:
        st.header('Forecast Controls')
        selected_country = st.selectbox('Country', countries, index=countries.index(default_country))
        selected_year = st.slider('Year', min_value=int(df['Year'].min()), max_value=2030, value=int(df['Year'].max()))
        show_top = st.checkbox('Show Top 10 Countries by Total Outflow', value=True)
        st.markdown('---')
        st.write('Model inputs are based on the selected country and year, with a country-specific historical median as a feature.')

    country_data = df[df['Country'] == selected_country].sort_values('Year')
    hist_median = float(country_data['Hist_Median'].iloc[0])
    predicted_outflow = predict_outflow(model, selected_year, hist_median)

    st.markdown('## Prediction Summary')
    col1, col2, col3 = st.columns([2, 2, 3])
    col1.metric('Country', selected_country)
    col2.metric('Forecast Year', selected_year)
    col3.metric('Predicted Outflow', f'{predicted_outflow:,.2f} USD Million')

    st.markdown('---')

    st.markdown('## Historical Trend & Forecast')
    fig_country = px.line(
        country_data,
        x='Year',
        y='Remittance_Outflow_USD_Mil',
        markers=True,
        title=f'{selected_country} Historical Outflows',
        labels={'Remittance_Outflow_USD_Mil': 'Outflow (USD Millions)'},
        template='plotly_white',
    )
    fig_country.update_layout(title=dict(x=0.02))

    forecast_color = 'red' if selected_year not in country_data['Year'].values else 'green'
    fig_country.add_scatter(
        x=[selected_year],
        y=[predicted_outflow],
        mode='markers+text',
        marker=dict(color=forecast_color, size=14),
        text=[f'{predicted_outflow:,.2f}'],
        textposition='top center',
        name='Forecast',
    )

    st.plotly_chart(fig_country, use_container_width=True)

    if show_top:
        st.markdown('## Top 10 Countries by Total Remittance Outflow')
        top_countries = (
            df.groupby('Country')['Remittance_Outflow_USD_Mil']
            .sum()
            .reset_index()
            .sort_values('Remittance_Outflow_USD_Mil', ascending=False)
            .head(10)
        )
        fig_top = px.bar(
            top_countries,
            x='Remittance_Outflow_USD_Mil',
            y='Country',
            orientation='h',
            title='Top 10 Countries by Cumulative Remittance Outflow',
            labels={'Remittance_Outflow_USD_Mil': 'Total Outflow (USD Millions)'},
            template='plotly_white',
        )
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, title=dict(x=0.02))
        st.plotly_chart(fig_top, use_container_width=True)

    st.markdown('## Model Insights')
    st.markdown(
        'This application uses a trained Random Forest regressor to forecast country outflows. ' 
        'The model learns from year and country historical median outflow patterns, making it robust to non-linear trends and high-value countries.'
    )
    st.markdown(
        '''
- **Non-linear forecasting**: Random Forest handles skewed financial data better than simple linear models.
- **Country-specific context**: Each forecast uses the historical median outflow for the chosen country.
- **Future-ready**: You can compare known historical years with future year forecasts up to 2030.
        '''
    )

    st.markdown('---')
    with st.expander('Data Notes'):
        st.write(
            '''
- Dataset range: 2000 through 2023.
- The app predicts in USD millions.
- Predictions for years beyond 2023 are driven by historical patterns and should be used as directional forecasts, not exact values.
            '''
        )


if __name__ == '__main__':
    main()
