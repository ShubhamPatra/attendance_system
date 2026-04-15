# Kubernetes Baseline Deployment

This directory provides a minimal Kubernetes baseline for AutoAttendance.

## Components
- `web` deployment/service (Flask + Gunicorn)
- `celery-worker` deployment
- `celery-beat` deployment
- `redis` deployment/service
- `ingress` for Nginx ingress controller compatible clusters

## Secrets and Config
Create secrets before applying manifests:

```bash
kubectl create secret generic autoattendance-secrets \
  --from-literal=MONGO_URI='...' \
  --from-literal=SECRET_KEY='...'
```

Apply manifests:

```bash
kubectl apply -f deploy/k8s/
```

Validate:

```bash
kubectl get pods,svc,ingress
kubectl logs deployment/autoattendance-web
```
