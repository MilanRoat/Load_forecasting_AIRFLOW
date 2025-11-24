to create user in AIrflow "airflow users create \
  --username admin \
  --password admin \
  --firstname admin \
  --lastname user \
  --role Admin \
  --email admin@example.com"
  run in "docker exec -it airflow_stack bash
"

run it using docker-compose up -d (change username and password in infrastructure/docker-compose )