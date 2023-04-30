FILE="Occupancy_corrupted.csv"

# set the URL of the web server
URL="http://localhost:8000/api/predict/temperature"

# read the CSV file line by line
while read -r line; do
  # send the line to the web server using curl
  curl -X POST -d "data=$line" "$URL"
done < "$FILE"
