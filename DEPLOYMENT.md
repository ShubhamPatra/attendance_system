# AutoAttendance Deployment Guide

## Local Production-Like (Docker Compose)

1. Copy `.env.example` to `.env` and set values:

- `MONGO_URI`
- `SECRET_KEY`
- `ENABLE_RBAC=1`
- `STUDENT_APP_PORT=5001`

1. Start services:

```bash
docker compose up -d --build
```

1. Access:

- App via Nginx: `http://localhost`
- Student portal: `http://localhost/student/`
- Legacy API docs: `http://localhost/api/docs`
- RESTX v2 docs (if `ENABLE_RESTX_API=1`): `http://localhost/api/v2/docs`

1. Health checks:

- `GET /health`
- `GET /ready`

## VM + Nginx

1. Run admin app behind Gunicorn:

```bash
gunicorn --config gunicorn.conf.py "app:create_app()"
```

1. Run student portal behind Gunicorn:

```bash
gunicorn --bind 0.0.0.0:5001 --config gunicorn.conf.py "student_app.app:create_app()"
```

1. Use Nginx config from `deploy/nginx/default.conf`.
1. For HTTPS, terminate TLS at Nginx with certbot-managed certs.

## Kubernetes Baseline

Use manifests under `deploy/k8s`.

```bash
kubectl create secret generic autoattendance-secrets \
  --from-literal=MONGO_URI='...' \
  --from-literal=SECRET_KEY='...'
kubectl apply -f deploy/k8s/autoattendance.yaml
```

## First Admin Bootstrap

Create initial admin user:

```bash
python scripts/bootstrap_admin.py --username admin --role admin
```
