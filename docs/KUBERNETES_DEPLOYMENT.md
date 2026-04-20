# Kubernetes Deployment Guide - AutoAttendance System

## Phase 7: Production Deployment with Health Probes

This guide covers deploying the AutoAttendance system to Kubernetes with production-ready health checks and monitoring.

---

## Prerequisites

- Kubernetes 1.19+ cluster (1.21+ recommended)
- kubectl configured for your cluster
- Docker image built: `attendance-admin:latest`
- MongoDB Atlas instance (or on-cluster MongoDB StatefulSet)
- NGINX Ingress Controller (recommended)
- Cert-Manager (for TLS certificates)
- Prometheus/Grafana (optional, for monitoring)

---

## Quick Start (5 minutes)

### Step 1: Configure Secrets

Edit `deploy/k8s/configmap.yaml` and set your actual values:

```bash
# Replace MongoDB connection string
# Replace with your domain in Ingress rules
# Update SECRET_KEY with a secure value
```

Create secrets:
```bash
kubectl create secret generic attendance-secrets \
  -n attendance-system \
  --from-literal=MONGO_URI='mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority' \
  --from-literal=SECRET_KEY='your-secure-key-here'
```

### Step 2: Deploy

```bash
# Create namespace
kubectl create namespace attendance-system

# Apply ConfigMap
kubectl apply -f deploy/k8s/configmap.yaml

# Apply Deployment, Service, NetworkPolicy
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml

# Verify deployment
kubectl get pods -n attendance-system
kubectl logs -f -n attendance-system deployment/attendance-admin
```

### Step 3: Verify Health

```bash
# Port forward to test locally
kubectl port-forward -n attendance-system svc/attendance-admin 8080:80

# Test health endpoints
curl http://localhost:8080/api/health
curl http://localhost:8080/api/readiness
curl http://localhost:8080/api/health/detailed
```

---

## Health Probe Configuration

### Overview

Three types of health checks ensure reliability:

1. **Startup Probe** (35s timeout)
   - Gives app time to initialize
   - Blocks liveness checks until successful
   
2. **Liveness Probe** (every 10s)
   - Endpoint: `GET /api/health`
   - Restarts pod if unhealthy for 30s
   - Monitors basic operation
   
3. **Readiness Probe** (every 5s)
   - Endpoint: `GET /api/readiness`
   - Removes from service if not ready
   - Checks full system readiness

### Probe Details

| Metric | Value | Purpose |
|--------|-------|---------|
| Startup initialDelaySeconds | 5 | Wait before first check |
| Startup periodSeconds | 5 | Check every 5s |
| Startup timeoutSeconds | 3 | HTTP timeout |
| Startup failureThreshold | 6 | 30s total (6×5) |
| **Liveness initialDelaySeconds** | **15** | **Wait 15s before monitoring** |
| **Liveness periodSeconds** | **10** | **Check every 10s** |
| **Liveness timeoutSeconds** | **5** | **HTTP timeout** |
| **Liveness failureThreshold** | **3** | **Restart after 30s** |
| **Readiness initialDelaySeconds** | **10** | **Ready check starts 10s** |
| **Readiness periodSeconds** | **5** | **Check every 5s** |
| **Readiness timeoutSeconds** | **3** | **HTTP timeout** |
| **Readiness failureThreshold** | **2** | **Remove from service after 10s** |

### Health Check Endpoints

#### `GET /api/health` (Liveness)

Returns simple status for Kubernetes liveness probe.

**Response (200 OK):**
```json
{
  "status": "ok"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "error"
}
```

**Checks:**
- ✓ Application responding

**Scenarios:**
- 200 OK: Pod is alive
- 503: Pod crashed or hung

---

#### `GET /api/health/detailed` (Manual Diagnostics)

Returns detailed health information for manual debugging.

**Response (200 OK):**
```json
{
  "status": "ok",
  "database": {
    "connected": true,
    "latency_ms": 12
  },
  "camera_status": "ready",
  "models_loaded": true,
  "metrics_engine": "running",
  "security_logs": "active"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "degraded",
  "database": {
    "connected": false,
    "error": "connection timeout"
  },
  "camera_status": "initializing",
  "models_loaded": false,
  "metrics_engine": "running",
  "reason": "Models not loaded yet"
}
```

**Use For:**
- Manual troubleshooting
- Operations dashboards
- Monitoring alerts

---

#### `GET /api/readiness` (Readiness)

Returns detailed readiness status for Kubernetes readiness probe.

**Response (200 OK - Ready):**
```json
{
  "ready": true,
  "database": "connected",
  "models": "loaded",
  "camera_pipeline": "initialized",
  "metrics_engine": "active"
}
```

**Response (503 Service Unavailable - Not Ready):**
```json
{
  "ready": false,
  "reason": "Models not loaded",
  "estimated_ready_in_seconds": 5,
  "database": "connected",
  "models": "loading",
  "camera_pipeline": "initializing",
  "metrics_engine": "starting"
}
```

**Readiness Conditions:**
- ✓ Database connection active
- ✓ At least 1 model loaded
- ✓ Camera pipeline initialized
- ✓ Metrics engine running

**Key Difference from Liveness:**
- Liveness: Is the pod alive?
- Readiness: Should traffic be sent to this pod?

---

## Configuration

### Environment Variables (ConfigMap)

See `deploy/k8s/configmap.yaml` for complete list. Key production settings:

```yaml
# Phase 1: Reliability
GRACEFUL_DEGRADATION_ENABLED: "1"
GRACEFUL_DEGRADATION_CPU_THRESHOLD: "80.0"
GRACEFUL_DEGRADATION_MEMORY_THRESHOLD: "85.0"

# Phase 2: Metrics
SLOW_FRAME_THRESHOLD_MS: "100.0"

# Phase 3: Confidence
RECOGNITION_TOP2_SIMILARITY_MARGIN: "0.05"
COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT: "0.5"
```

### Secrets

Create Kubernetes secret before deploying:

```bash
kubectl create secret generic attendance-secrets \
  -n attendance-system \
  --from-literal=MONGO_URI='<your-connection-string>' \
  --from-literal=SECRET_KEY='<secure-random-key>'
```

---

## Resource Allocation

### CPU and Memory

| Resource | Request | Limit | Rationale |
|----------|---------|-------|-----------|
| **CPU Request** | 500m | 2000m | Face detection needs CPU |
| **Memory Request** | 1Gi | 2Gi | Models + frame buffers |

**Recommendations:**
- **High-throughput** (500+ students/hour): Increase to 1000m CPU, 2Gi RAM
- **Multi-camera** (3+): Increase to 2000m CPU, 4Gi RAM
- **Edge/limited**: Decrease to 200m CPU, 512Mi RAM

### Storage

```yaml
volumes:
- name: logs
  emptyDir:
    sizeLimit: 500Mi
- name: tmp
  emptyDir:
    sizeLimit: 1Gi
```

**Notes:**
- Logs are in-memory only; configure external logging (ELK/DataDog) for persistence
- Temporary storage for frame processing

---

## Scaling

### Horizontal Pod Autoscaling (HPA)

Configured in `service.yaml` to scale 2-5 replicas based on:

- **CPU**: Scale up at 70% utilization
- **Memory**: Scale up at 80% utilization

**Configuration:**
```yaml
minReplicas: 2
maxReplicas: 5
scaleUp: Aggressive (100% increase per 30s)
scaleDown: Conservative (50% decrease per 60s)
```

**Monitoring Autoscaling:**
```bash
kubectl get hpa -n attendance-system -w
kubectl describe hpa attendance-admin-hpa -n attendance-system
```

---

## Networking

### Service Types

- **Service (ClusterIP)**: Internal only, port 80 → 5000
- **Ingress**: External via HTTPS (requires cert-manager)
- **NetworkPolicy**: Restrict traffic to/from pod

### NetworkPolicy Rules

```yaml
Ingress:
- From nginx-ingress-controller (port 5000)
- From monitoring namespace (Prometheus)

Egress:
- To kube-dns (port 53, UDP) - DNS lookups
- To MongoDB Atlas (ports 27017-20, 443) - Database
- To any pod in namespace (port 5000)
```

**To disable NetworkPolicy:**
```bash
kubectl delete networkpolicy -n attendance-system attendance-admin-network-policy
```

---

## Ingress Setup

### Prerequisites

1. NGINX Ingress Controller
   ```bash
   helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
   helm install nginx-ingress ingress-nginx/ingress-nginx \
     -n ingress-nginx --create-namespace
   ```

2. Cert-Manager
   ```bash
   helm repo add jetstack https://charts.jetstack.io
   helm install cert-manager jetstack/cert-manager \
     --namespace cert-manager --create-namespace \
     --set installCRDs=true
   ```

### Configuration

Edit `deploy/k8s/service.yaml`:

```yaml
spec:
  rules:
  - host: "attendance-admin.your-domain.com"  # ← Change this
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: attendance-admin
            port:
              number: 80
```

Apply:
```bash
kubectl apply -f deploy/k8s/service.yaml
```

---

## Monitoring & Observability

### Health Check Status

```bash
# Check pod status
kubectl get pods -n attendance-system

# Check probe results in events
kubectl describe pod -n attendance-system <pod-name>

# Check recent probe failures
kubectl get events -n attendance-system --sort-by='.lastTimestamp'
```

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Startup probe fails | App takes >35s | Increase startupProbe.failureThreshold |
| Liveness restarts pod | App timeout | Increase liveness timeoutSeconds |
| Readiness not ready | Models not loaded | Check logs for model errors |
| Cannot connect to DB | Network/credentials | Verify MONGO_URI secret |

### Viewing Logs

```bash
# Real-time logs
kubectl logs -f -n attendance-system deployment/attendance-admin

# Logs from specific pod
kubectl logs -n attendance-system <pod-name>

# Previous pod logs (if crashed)
kubectl logs -n attendance-system <pod-name> --previous
```

### Prometheus Integration

For Prometheus scraping metrics:

```bash
kubectl apply -f docs/kubernetes/prometheus-service-monitor.yaml
```

Access metrics at: `http://localhost:5000/metrics` (requires port-forward)

---

## Updating & Rollbacks

### Rolling Update

```bash
# Update image (triggers rolling update due to RollingUpdate strategy)
kubectl set image deployment/attendance-admin \
  admin=attendance-admin:v2.0 \
  -n attendance-system

# Watch rollout progress
kubectl rollout status deployment/attendance-admin -n attendance-system
```

### Rollback

```bash
# Show rollout history
kubectl rollout history deployment/attendance-admin -n attendance-system

# Rollback to previous version
kubectl rollout undo deployment/attendance-admin -n attendance-system

# Rollback to specific revision
kubectl rollout undo deployment/attendance-admin -n attendance-system --to-revision=3
```

---

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod -n attendance-system <pod-name>

# Check logs
kubectl logs -n attendance-system <pod-name>

# Common issues:
# - Image not found: Verify image registry
# - PullBackOff: Check image credentials
# - Pending: Check node resources (kubectl top nodes)
```

### Health Check Fails

```bash
# Connect to pod shell
kubectl exec -it -n attendance-system <pod-name> -- /bin/bash

# Test health endpoint inside pod
curl http://localhost:5000/api/health
curl http://localhost:5000/api/readiness

# Check app logs
tail -f logs/app.log
```

### Database Connection Issues

```bash
# Verify MONGO_URI is set
kubectl get secret -n attendance-system attendance-secrets -o jsonpath='{.data.MONGO_URI}' | base64 -d

# Check database connectivity from pod
kubectl exec -it -n attendance-system <pod-name> -- python -c \
  "from pymongo import MongoClient; print(MongoClient('<MONGO_URI>').admin.command('ping'))"
```

---

## Production Checklist

- [ ] MONGO_URI and SECRET_KEY set in secrets
- [ ] Domain configured in Ingress (service.yaml)
- [ ] TLS certificates ready (cert-manager)
- [ ] Resource requests/limits verified for your load
- [ ] NetworkPolicy reviewed and applied
- [ ] RBAC roles configured (ServiceAccount, Role, RoleBinding)
- [ ] HPA configured for expected scaling
- [ ] PDB (Pod Disruption Budget) reviewed
- [ ] Monitoring/logging setup (Prometheus, ELK, etc.)
- [ ] Backup strategy for persistent data
- [ ] Disaster recovery plan documented
- [ ] Network policies tested

---

## Advanced: Multi-Region Deployment

For geographic redundancy:

1. Deploy to multiple clusters
2. Use MongoDB Atlas Global Clusters
3. Configure DNS load balancing (Route53, Google Cloud DNS)
4. Implement cross-cluster health checks

---

## References

- [Kubernetes Health Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [Cert-Manager](https://cert-manager.io/)
- [HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/)

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs: `kubectl logs -f -n attendance-system deployment/attendance-admin`
3. Test health endpoints manually
4. Open issue with:
   - Pod status: `kubectl describe pod -n attendance-system <pod-name>`
   - Recent events: `kubectl get events -n attendance-system --sort-by='.lastTimestamp'`
   - App logs: `kubectl logs -n attendance-system <pod-name>`
