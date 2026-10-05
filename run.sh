#!/bin/bash
cd /home/btcsunrise/projects/btc-sim-echophase/
poetry run python src/simulate.py --iterations 10000 --data data/btc_daily_price.csv --output output/monthly_forecast.csv
