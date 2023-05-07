#!/usr/bin/bash
python3 clone.py
rm myapp/lstm_model_temperature_9.pt
rm tempNewData.csv
rm h_tensor.pt
rm temperature_data.npz
python3 manage.py runserver 0.0.0.0:8000
