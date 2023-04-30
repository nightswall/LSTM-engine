#!/bin/bash

FILE="Occupancy.csv"

# set the URL of the web server
URL="http://localhost:8000/api/predict/temperature"

# initialize a counter variable
counter=0

# read the CSV file line by line
while read -r line; do
  # increment the counter
  ((counter++))

  # check if the counter is divisible by 90
  if ((counter % 15 == 0)); then
    # send the line to the web server using curl
    curl -X POST -d "data=$line" "$URL"
  fi
done < "$FILE"

