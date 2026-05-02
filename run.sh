#!/usr/bin/env bash
# QUANT LAB launcher — opens the Streamlit app
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py "$@"
