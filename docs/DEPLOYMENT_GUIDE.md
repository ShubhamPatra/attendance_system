# Production Deployment Guide

**Target Audience**: DevOps engineers, system administrators, deployment specialists  
**Prerequisite Reading**: [SETUP_DETAILED.md](IMPLEMENTATION/SETUP_DETAILED.md), [ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md)  
**Last Updated**: April 20, 2026

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Topologies](#deployment-topologies)
3. [Production Configuration](#production-configuration)
4. [Monitoring & Alerting Setup](#monitoring--alerting-setup)
5. [Backup & Disaster Recovery](#backup--disaster-recovery)
6. [Upgrading & Rolling Updates](#upgrading--rolling-updates)
7. [Troubleshooting Production Issues](#troubleshooting-production-issues)
8. [Security Hardening](#security-hardening)

---

## Pre-Deployment Checklist

### Infrastructure Validation

```yaml
Checklist:
  Hardware:
    - ✓ CPU: Sufficient cores (4+ for small, 16+ for medium, 32+ for enterprise)
    - ✓ RAM: 8GB min, 32GB recommended, 128GB for large deployments
    - ✓ Storage: SSD primary (fast boot), HDD backup (large capacity)
    - ✓ Network: Gigabit minimum, 10Gbps recommended for high throughput
    - ✓ UPS/Power: Redundant power supplies, uninterruptible backup
  
  Networking:
    - ✓ Firewall rules: Allow 80 (HTTP), 443 (HTTPS), 27017 (MongoDB), 6379 (Redis)
    - ✓ DNS: Correct A records pointing to load balancer
    - ✓ SSL/TLS: Valid certificates, not expired, chain complete
    - ✓ Bandwidth: Reserve 2-5 Mbps per concurrent user
  
  Database:
    - ✓ MongoDB: Replica set initialized (primary + 2 secondaries)
    - ✓ Sharding: Configured (if >10K students)
    - ✓ Indexes: Created (see DATABASE_DESIGN.md)
    - ✓ Backups: Automated schedule configured
    - ✓ User accounts: Created with appropriate permissions
  
  Application:
    - ✓ Models downloaded: YuNet, ArcFace, anti-spoofing
    - ✓ FAISS index: Generated from enrolled students
    - ✓ Configuration: Tuned for target scale (thresholds, timeouts)
    - ✓ Secrets: API keys, JWT secrets stored in secure vault
    - ✓ Logging: Log aggregation service configured
```

### Pre-Flight Tests

```bash
# 1. Connectivity tests
ping database-primary.internal
ping redis-node.internal
curl https://attendance-api.school.edu/health

# 2. Model inference test
python scripts/test_models.py
# Output: YuNet: OK (33ms), ArcFace: OK (18ms), Liveness: OK (20ms)

# 3. Load test (simulate 50 concurrent students)
python scripts/load_test.py --concurrent=50 --duration=60s
# Output: Throughput: 500 faces/min, P95: 120ms, Errors: 0

# 4. Failover test
# Kill MongoDB primary → verify automatic failover
# Kill one Flask instance → verify load balancer redirect
# Verify detection of failures within 30 seconds
```

---

## Deployment Topologies

### Topology 1: Single Server (100-500 students)

**Architecture**:
```
┌─────────────────────────────────────┐
│         Single Server               │
├─────────────────────────────────────┤
│  Flask App (Gunicorn, 4 workers)   │
│  YuNet, ArcFace, Liveness (all CPU)│
│  MongoDB (single instance)          │
│  Redis (in-memory cache)            │
└─────────────────────────────────────┘
         ↓
    Clients (Web/Mobile)
```

**Deployment Steps**:

1. **Install Dependencies**:
```bash
sudo apt update
sudo apt install -y python3.9 python3-pip mongodb redis-server
pip install -r requirements.txt
```

2. **Configure MongoDB**:
```bash
# Edit /etc/mongod.conf
sudo nano /etc/mongod.conf

# Uncomment/set:
# port: 27017
# dbPath: /var/lib/mongodb
# journal:
#   enabled: true

sudo systemctl start mongod
sudo systemctl enable mongod
```

3. **Start Application**:
```bash
# Create systemd service
sudo tee /etc/systemd/system/attendance-app.service > /dev/null <<EOF
[Unit]
Description=AutoAttendance Flask Application
After=mongod.service redis-server.service

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/attendance
ExecStart=gunicorn --workers=4 --bind=127.0.0.1:5000 --timeout=30 app:app
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable attendance-app
sudo systemctl start attendance-app
```

**Verification**:
```bash
curl -H "Authorization: Bearer <token>" https://attendance.school.edu/api/health
# Output: {"status": "healthy", "database": true, "models": true}
```

---

### Topology 2: Load-Balanced Cluster (500-2000 students)

**Architecture**:
```
                    Load Balancer (HAProxy/AWS ELB)
                    /          |          \
              Gunicorn-1   Gunicorn-2   Gunicorn-3
              (4 workers) (4 workers) (4 workers)
                    \          |          /
                          ↓
                MongoDB Replica Set
                (Primary + 2 Replicas)
                          ↓
                    Redis Cache
                  (1 master node)
```

**Configuration**:

1. **MongoDB Replica Set**:
```bash
# On Primary (mongo-1)
mongo
> rs.initiate({
  _id: "rs0",
  members: [
    {_id: 0, host: "mongo-1.internal:27017", priority: 2},
    {_id: 1, host: "mongo-2.internal:27017", priority: 1},
    {_id: 2, host: "mongo-3.internal:27017", priority: 1}
  ]
})

# Verify
> rs.status()
# Output: All 3 nodes SECONDARY, Primary elected within 5 seconds
```

2. **Nginx Load Balancing** (Optional - if using Nginx):
```nginx
upstream flask_cluster {
    least_conn;  # Route to least-connected server
    
    server flask-1.internal:5000 max_fails=3 fail_timeout=30s;
    server flask-2.internal:5000 max_fails=3 fail_timeout=30s;
    server flask-3.internal:5000 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    
    # Session stickiness (route same client to same backend)
    map $remote_addr $backend {
        default "flask_cluster";
    }
    
    location / {
        proxy_pass http://$backend;
        proxy_read_timeout 10s;
        proxy_connect_timeout 5s;
    }
}
```

**Note**: Alternatively, use a cloud load balancer (AWS ELB, GCP Load Balancer, Azure LB) instead of Nginx.

2. **Flask Configuration** (on each instance):
```python
# config.py
class ProductionConfig:
    # Database
    MONGO_URI = "mongodb://mongo-1.internal:27017,mongo-2.internal:27017,mongo-3.internal:27017/?replicaSet=rs0"
    
    # Redis (single node, but can upgrade to cluster)
    REDIS_URL = "redis://redis-1.internal:6379/0"
    
    # Session management
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Performance
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB file upload max
    SEND_FILE_MAX_AGE_DEFAULT = 3600  # 1 hour cache
    
    # Timeouts
    PERMANENT_SESSION_LIFETIME = 7200  # 2 hours
```

---

### Topology 3: Kubernetes Enterprise (2000+ students)

**Architecture**:
```
                    Kubernetes Ingress
                    /  |  |  |  \
            Pod-1  Pod-2  ...  Pod-N  (Auto-scaling 3-20 replicas)
                \       |       /
                StatefulSet: MongoDB Sharded Cluster
                ├─ Shard 1 (mongo-shard-1-0, mongo-shard-1-1, mongo-shard-1-2)
                ├─ Shard 2 (mongo-shard-2-0, mongo-shard-2-1, mongo-shard-2-2)
                └─ Config Server (mongo-config-0, mongo-config-1, mongo-config-2)
                
                StatefulSet: Redis Cluster
                ├─ Node 1 (Primary)
                ├─ Node 2 (Replica)
                └─ Node 3-6 (Additional shards)
                
                PersistentVolumes: Model storage, backups
```

**Deployment YAML**:
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: flask-app
  template:
    metadata:
      labels:
        app: flask-app
    spec:
      containers:
      - name: flask
        image: attendance:latest
        ports:
        - containerPort: 5000
        env:
        - name: MONGO_URI
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: mongodb-uri
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: flask-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: flask-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

**Deployment Command**:
```bash
kubectl apply -f kubernetes/
kubectl rollout status deployment/flask-app --namespace=production
kubectl get hpa flask-hpa --namespace=production --watch
```

---

## Production Configuration

### Environment Variables

```bash
# .env.production
# Database
MONGO_URI=mongodb://mongo-rs/attendance?replicaSet=rs0&authSource=admin
MONGO_USERNAME=app_user
MONGO_PASSWORD=<secure-password>

# Cache
REDIS_URL=redis://redis-cluster:6379/0
REDIS_PASSWORD=<secure-password>

# Application
FLASK_ENV=production
FLASK_DEBUG=false
SECRET_KEY=<64-char-random-string>
JWT_SECRET=<64-char-random-string>

# Models
YUNET_MODEL_PATH=/opt/models/face_detection_yunet.onnx
ARCFACE_MODEL_PATH=/opt/models/arcface_resnet100.onnx
SPOOF_MODEL_PATH=/opt/models/silent_face.onnx
FAISS_INDEX_PATH=/opt/models/faiss_index.bin

# Performance
MAX_WORKERS=4
BATCH_SIZE=16
GPU_ENABLED=true  # Set to false for CPU-only deployments

# Security
ALLOWED_ORIGINS=https://attendance.school.edu
CORS_CREDENTIALS=true
RATE_LIMIT=1000  # requests per hour

# Monitoring
LOG_LEVEL=INFO
SENTRY_DSN=https://<sentry-dsn>
DATADOG_API_KEY=<datadog-api-key>
```

### Health Check Endpoints

```python
@app.route('/health', methods=['GET'])
def health():
    """Liveness probe - basic health check"""
    return jsonify({'status': 'alive'}), 200

@app.route('/ready', methods=['GET'])
def ready():
    """Readiness probe - can handle traffic?"""
    checks = {
        'database': check_mongo_connectivity(),
        'cache': check_redis_connectivity(),
        'models': check_models_loaded(),
        'disk': check_disk_space(min_gb=5)
    }
    
    all_ready = all(checks.values())
    status = 200 if all_ready else 503
    return jsonify({'ready': all_ready, 'checks': checks}), status

def check_mongo_connectivity():
    try:
        db.command('ping')
        return True
    except Exception:
        return False
```

---

## Monitoring & Alerting Setup

### Prometheus Metrics Collection

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'flask-app'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 5s
  
  - job_name: 'mongodb'
    static_configs:
      - targets: ['localhost:27017']
    metrics_path: '/metrics'
  
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:6379']
```

### Key Alerts

```yaml
# prometheus/alerts.yml
groups:
- name: application_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    annotations:
      summary: "High error rate detected ({{ $value }})"
  
  - alert: HighLatency
    expr: histogram_quantile(0.95, request_duration_seconds) > 0.5
    for: 5m
    annotations:
      summary: "P95 latency above 500ms ({{ $value }}s)"
  
  - alert: LowAccuracy
    expr: face_detection_accuracy{status="found"} / (face_detection_accuracy{status="found"} + face_detection_accuracy{status="not_found"}) < 0.98
    for: 1h
    annotations:
      summary: "Detection accuracy below 98% threshold"
  
  - alert: DatabaseConnectivityIssue
    expr: up{job="mongodb"} == 0
    for: 1m
    annotations:
      summary: "MongoDB is down"
  
  - alert: OutOfDiskSpace
    expr: disk_free_bytes / disk_total_bytes < 0.1
    for: 5m
    annotations:
      summary: "Less than 10% disk space remaining"
```

### Log Aggregation

```python
# Centralized logging with ELK stack
import logging
from elasticsearch import Elasticsearch
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Log entries automatically sent to Elasticsearch via Filebeat
logger.info("Attendance marked", extra={
    'student_id': 'STU2025001',
    'confidence': 0.92,
    'latency_ms': 103,
    'timestamp': datetime.now().isoformat()
})
```

---

## Backup & Disaster Recovery

### Automated Backup Strategy

**MongoDB Backup** (daily at 2 AM):
```bash
#!/bin/bash
# /usr/local/bin/backup-mongodb.sh

BACKUP_DIR="/backups/mongodb"
DATE=$(date +%Y-%m-%d)
BACKUP_PATH="$BACKUP_DIR/mongodb-$DATE"

# Create backup
mongodump --uri="mongodb://localhost:27017" --out="$BACKUP_PATH"

# Compress
tar -czf "$BACKUP_PATH.tar.gz" "$BACKUP_PATH"

# Upload to S3 (AWS)
aws s3 cp "$BACKUP_PATH.tar.gz" "s3://attendance-backups/mongodb/$DATE/"

# Clean local backups older than 30 days
find "$BACKUP_DIR" -name "mongodb-*.tar.gz" -mtime +30 -delete

# Send status to monitoring
curl -X POST https://monitoring.school.edu/backup-status \
  -d "{\"status\": \"success\", \"size_gb\": $(du -sh $BACKUP_PATH | cut -f1)}"
```

**Cron Schedule**:
```bash
# /etc/cron.d/attendance-backup
0 2 * * * /usr/local/bin/backup-mongodb.sh >> /var/log/backup.log 2>&1
0 3 * * 0 /usr/local/bin/verify-backup.sh >> /var/log/backup.log 2>&1
```

### Disaster Recovery Procedure

**Scenario 1: Database Corruption**

```bash
# 1. Stop application
sudo systemctl stop attendance-app

# 2. Restore from latest backup
BACKUP_DATE=$(date -d "1 day ago" +%Y-%m-%d)
aws s3 cp "s3://attendance-backups/mongodb/$BACKUP_DATE/mongodb-$BACKUP_DATE.tar.gz" /tmp/

tar -xzf "/tmp/mongodb-$BACKUP_DATE.tar.gz" -C /tmp/

# 3. Restore to MongoDB
mongorestore --drop /tmp/mongodb-$BACKUP_DATE/

# 4. Verify data integrity
mongo attendance --eval "db.students.count(); db.attendance.count()"

# 5. Restart application
sudo systemctl start attendance-app

# 6. Run full health check
curl https://attendance.school.edu/health
```

**RTO/RPO Analysis**:
- Recovery Time Objective (RTO): 30 minutes
- Recovery Point Objective (RPO): 24 hours (last backup)

---

## Upgrading & Rolling Updates

### Blue-Green Deployment Strategy

```bash
#!/bin/bash
# deploy-new-version.sh

NEW_VERSION=$1
CURRENT_VERSION=$(cat /opt/attendance/VERSION)

echo "Deploying $NEW_VERSION (current: $CURRENT_VERSION)"

# 1. Create new version directory
mkdir -p /opt/attendance/v$NEW_VERSION
cd /opt/attendance/v$NEW_VERSION
git clone --branch $NEW_VERSION https://github.com/ShubhamPatra/attendance_system.git .
pip install -r requirements.txt

# 2. Create systemd service for green environment
sudo tee /etc/systemd/system/attendance-app-green.service > /dev/null <<EOF
[Unit]
Description=AutoAttendance Flask Application (Green)
After=mongod.service redis-server.service

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/attendance/v$NEW_VERSION
ExecStart=gunicorn --workers=4 --bind=127.0.0.1:5001 --timeout=30 app:app
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

# 3. Start green environment
sudo systemctl daemon-reload
sudo systemctl start attendance-app-green
sleep 5

# 4. Health check green environment
HEALTH=$(curl -s http://localhost:5001/api/health | jq -r '.status')
if [ "$HEALTH" != "healthy" ]; then
  echo "Green environment failed health check"
  sudo systemctl stop attendance-app-green
  exit 1
fi

# 5. Load test green environment
python3 scripts/load_test.py --target=http://localhost:5001 --concurrent=50
if [ $? -ne 0 ]; then
  echo "Load test failed on green environment"
  sudo systemctl stop attendance-app-green
  exit 1
fi

# 6. Switch traffic to green (update load balancer config or use systemd service alias)
# If using systemd:
sudo systemctl stop attendance-app  # Stop blue (old version)
sudo systemctl start attendance-app-green  # Start green (new version)
# Rename green to standard service
sudo mv /etc/systemd/system/attendance-app-green.service /etc/systemd/system/attendance-app.service

# 7. Monitor for errors (5 minutes)
for i in {1..60}; do
  ERROR_RATE=$(curl -s http://localhost:5000/api/metrics | jq -r '.error_rate // 0')
  if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
    echo "Error rate spike detected, rolling back"
    # Rollback: restore old service and stop green
    sudo systemctl stop attendance-app
    exit 1
  fi
  sleep 5
done

# 8. Update version file
echo "$NEW_VERSION" | sudo tee /opt/attendance/VERSION
echo "Deployment successful"
```

**Rollback Procedure** (if issues detected):
```bash
# Immediate rollback to previous version
sudo systemctl stop attendance-app
cd /opt/attendance
git checkout $PREVIOUS_VERSION
sudo systemctl start attendance-app
curl https://attendance.school.edu/api/health  # Verify health
```

---

## Troubleshooting Production Issues

### Issue: High Latency (>500ms per request)

**Diagnosis**:
```bash
# 1. Check application metrics
curl https://attendance.school.edu/metrics | grep request_duration

# 2. Check database performance
mongo attendance --eval "
  db.system.profile.find().limit(5).pretty()
"

# 3. Check GPU utilization
nvidia-smi

# 4. Check network latency
ping -c 5 mongo-1.internal
```

**Solutions**:
```
If GPU underutilized:
  → Enable batch processing (BATCH_SIZE=32)
  → Reduce model precision (FP32 → INT8)

If database slow:
  → Check indexes exist
  → Verify query plans with explain()
  → Increase MongoDB cache (wiredTiger cache)

If network high latency:
  → Move services closer (same datacenter)
  → Enable connection pooling
  → Use local FAISS index cache
```

### Issue: Memory Leak

**Detection**:
```bash
# Monitor resident memory over time
watch -n 5 'ps aux | grep flask'

# If memory grows continuously:
# Check for unclosed database connections
# Check for large objects held in Python cache
```

**Fix**:
```python
# Ensure connections are closed
@app.teardown_appcontext
def close_db(error):
    db_connection.close()
    redis_connection.close()

# Clear cache periodically
from cachetools import TTLCache
model_cache = TTLCache(maxsize=100, ttl=3600)  # 1-hour TTL
```

### Issue: Spoofing Detection Failing

**Analysis**:
```python
# Enable debug logging for liveness verification
import logging
logging.getLogger('liveness').setLevel(logging.DEBUG)

# Log each layer's detection result
logger.debug(f"CNN: {cnn_score}, Blink: {blink_score}, Motion: {motion_score}, Heuristics: {heur_score}")
logger.debug(f"Final score: {final_score}, Threshold: {threshold}, Result: {'PASS' if final_score > threshold else 'FAIL'}")
```

**Calibration Procedure**:
```bash
# Re-calibrate thresholds based on actual data
python scripts/calibrate_liveness_threshold.py \
  --training-data=/data/liveness_samples \
  --target-false-positive=0.02 \
  --target-false-negative=0.03

# Output: Recommended threshold = 0.52 (was 0.50)
```

---

## Security Hardening

### SSL/TLS Configuration

```nginx
# /etc/nginx/conf.d/ssl.conf
# Strong SSL configuration (A+ on SSL Labs)

ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;

# HSTS (enforce HTTPS for future visits)
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

# Prevent MIME type sniffing
add_header X-Content-Type-Options "nosniff" always;

# Clickjacking protection
add_header X-Frame-Options "SAMEORIGIN" always;

# Enable XSS filter
add_header X-XSS-Protection "1; mode=block" always;

# CSP (Content Security Policy)
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'" always;
```

### Database Access Control

```javascript
// MongoDB user permissions (least privilege)
use attendance
db.createUser({
  user: "app_user",
  pwd: "<strong-password>",
  roles: [
    { role: "read", db: "attendance" },
    { role: "readWrite", db: "attendance" }
  ]
})

db.createUser({
  user: "backup_user",
  pwd: "<strong-password>",
  roles: [
    { role: "backup", db: "admin" },
    { role: "restore", db: "admin" }
  ]
})
```

### API Security Headers

```python
# Add security headers to all responses
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response
```

---

## Maintenance Schedule

| Task | Frequency | Duration |
|------|-----------|----------|
| Database backup | Daily | 30 min |
| Security patches | Weekly | 1 hour |
| Certificate renewal | Quarterly | 15 min |
| Model retraining | Monthly | 4 hours |
| Full system test | Monthly | 2 hours |
| Disaster recovery drill | Quarterly | 2 hours |
| Performance tuning | Quarterly | 4 hours |

---

## References

- [ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md)
- [DATABASE_DESIGN.md](ALGORITHM_DEEP_DIVES/DATABASE_DESIGN.md)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MongoDB Replica Sets](https://docs.mongodb.com/manual/replication/)
- [Prometheus Monitoring](https://prometheus.io/)
