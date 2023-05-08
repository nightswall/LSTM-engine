# Project Title: GPU-LSTM Temperature Prediction

This project uses a GPU-accelerated Long Short-Term Memory (LSTM) model to predict temperature values based on historical data. The project is containerized using Docker for easy deployment and scalability.

## Project Structure

```
.
├── clone.py                        # Script to clone data from repository
├── db.sqlite3                      # SQLite database file
├── debug.py                        # Debugging script
├── Dockerfile                      # CPU-based Dockerfile
├── Dockerfile.gpu                  # GPU-based Dockerfile
├── lstm_model_temperature_10.pt    # Pretrained LSTM model
├── main.csv                        # Main CSV file containing historical data
├── manage.py                       # Django management script
├── myapp                           # Django app directory
├── myproject                       # Django project directory
├── no_anomaly.csv                  # CSV file containing cleaned data without anomalies
├── outputBus0.csv                  # CSV file containing output data
├── requirements.txt                # Python dependencies file
├── start.sh                        # Shell script to start the Django server
└── test                            # Test directory
```

## How to Build and Run the Docker Image

1. Make sure Docker and NVIDIA Container Toolkit are properly installed on your system.

2. Open a terminal and navigate to the project's root directory.

3. Build the GPU-based Docker image by running the following command:

```bash
docker build -t gpu-lstm -f Dockerfile.gpu .
```

4. Run the GPU-based Docker container by executing the following command:

```bash
docker run --gpus all -it --rm -p 80:8000 --security-opt seccomp=unconfined gpu-lstm:latest
```

The above command will run the Docker container and map the container's port 8000 to your host machine's port 80. You can then access the application by navigating to http://localhost:80 in your web browser.

Note: If you encounter any issues with GPU access or permissions, make sure to check that your NVIDIA drivers, Docker version, and NVIDIA Container Toolkit are up-to-date and properly installed.
