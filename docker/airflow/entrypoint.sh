#!/usr/bin/env bash
set -e

# Pre-seed fixed credentials for standalone mode and simple auth manager
mkdir -p /opt/airflow
echo 'admin123' > /opt/airflow/standalone_admin_password.txt
echo '{"admin": "admin123"}' > /opt/airflow/simple_auth_manager_passwords.json 2>/dev/null || true
echo '{"admin": "admin123"}' > /opt/airflow/simple_auth_manager_passwords.json.generated 2>/dev/null || true

exec /entrypoint standalone
