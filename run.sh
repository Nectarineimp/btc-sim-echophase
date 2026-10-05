#!/bin/bash
cd /home/btcsunrise/projects/btc-sim-echophase/
poretry run python src/simulate.py --iterations 10000 --data data/btc_daily.csv --output output/monthly_forecast.csv