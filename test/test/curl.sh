FILE="fivedayoutputtest.csv"

# set the URL of the web server
#URL="http://172.18.0.76:8001/api/predict/power" #for docker
URL="http://localhost:8001/api/predict/current" #for local

# read the CSV file line by line
while read -r line; do
  # send the line to the web server using curl
  curl -X POST -d "data=$line" "$URL"
  sleep 1
done < "$FILE"
