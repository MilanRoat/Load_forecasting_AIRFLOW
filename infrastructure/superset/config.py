# Set a custom SECRET_KEY
SECRET_KEY = 'fE/v+aG7oP3xR2bA8jV6kL9yU4wS5tE/qZ1dC0gH4nO9mK7pY+sB'

# Set the database location
SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://airflow:airflow@postgres:5432/superset'

# This helps Superset run in a container
TALISMAN_ENABLED = False