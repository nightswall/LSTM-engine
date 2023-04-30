#!/usr/bin/bash
rm tempNewData.csv
rm h_tensor.pt
rm temperature_data.npz
python3 manage.py runserver 0.0.0.0:8000
