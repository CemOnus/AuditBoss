# CB Ops — ISO Certification Body SaaS (Demo)

Streamlit app to manage ISO certification body operations (Quotes, Scheduling, Auditors, Certificates, Invoicing, Reports).

- Ships with demo CSVs and will also auto-generate data if files are missing.
- All date fields are normalized to Pandas Timestamp to avoid dtype comparison errors.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Streamlit Cloud
Upload repo with these three files and set entrypoint to app.py.