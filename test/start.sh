#!/usr/bin/bash
# Copy database file to buffer file 
#python3 clone.py
# Remove files from earlier runs
rm h_tensor.pt
#echo "updated"
#rm myapp/lstm_model_Temperature_9.pt
rm tempNewDataTemperature.csv
rm Temperature_data.npz
#rm myapp/lstm_model_Voltage_9.pt
rm tempNewDataVoltage.csv
rm Voltage_data.npz
#rm myapp/lstm_model_Power_9.pt
rm tempNewDataPower.csv
rm Power_data.npz
#rm myapp/lstm_model_Current_9.pt
rm tempNewDataCurrent.csv
rm Current_data.npz
# Run server
python3 manage.py runserver 0.0.0.0:8001 --noreload
