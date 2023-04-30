# Use an official Python runtime as the base image
FROM python:3.10.6

# Set the working directory in the container
WORKDIR /gru-lstm

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the application dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

RUN chmod +x start.sh
# Expose the port that the application will run on
# EXPOSE 8000

# Set the environment variable for the Django settings module
ENV DJANGO_SETTINGS_MODULE myproject.settings

# Run any necessary database migrations
RUN python manage.py migrate

# Collect static files
# RUN python manage.py collectstatic --noinput
# RUN ls -ahl
# Start the application
CMD [ "bash","start.sh" ] 
