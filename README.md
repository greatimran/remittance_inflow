# Remittance Outflow Predictor

A Streamlit web application that forecasts country-level remittance outflow in USD millions using a trained Random Forest model.

## Included Files
- `app.py` - Streamlit app source code
- `requirements.txt` - Python dependencies
- `refined_remittance_outflows.csv` - input dataset
- `remittance_model.pkl` - trained Random Forest model
- `train_and_save_model.py` - training script to regenerate the model
- `.gitignore` - ignores the virtual environment and temporary files
- `.streamlit/config.toml` - Streamlit theme and server configuration

## Run locally
1. Open a terminal in the project folder `D:/python_working/zafar_iqbal/Remittance_Inflow`
2. Create and activate a virtual environment if not already created:
   - PowerShell: `.
emittance_inflow\venv\Scripts\Activate.ps1`
   - CMD: `.
emittance_inflow\venv\Scripts\activate.bat`
3. Install dependencies:
   - `python -m pip install -r requirements.txt`
4. Start the app:
   - `streamlit run app.py`
5. Open the browser at `http://localhost:8501`

## Deploy on Streamlit Cloud
1. Push the project folder to a GitHub repository.
2. Ensure the repository includes:
   - `app.py`
   - `requirements.txt`
   - `refined_remittance_outflows.csv`
   - `remittance_model.pkl`
   - `.streamlit/config.toml`
3. Go to Streamlit Cloud and connect your GitHub repository.
4. Choose the repository branch and app file: `app.py`.
5. Deploy the app.

### If the model file is too large for GitHub
- Remove `remittance_model.pkl` from the repo before pushing.
- Add `train_and_save_model.py` to the repo.
- Create a lightweight `requirements.txt` and deploy.
- After deployment, you can run `train_and_save_model.py` on the host (if supported) to regenerate `remittance_model.pkl`.

## Notes
- The app uses the local dataset and model file stored in the same folder.
- `.gitignore` excludes `venv/`, so the virtual environment is not pushed to GitHub.
- Streamlit Cloud will install packages from `requirements.txt` automatically.
