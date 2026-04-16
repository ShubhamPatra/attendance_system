# Frequently Asked Questions (FAQ)

## Table of Contents

1. [General Questions](#general-questions)
2. [Technical Capability Questions](#technical-capability-questions)
3. [Deployment Questions](#deployment-questions)
4. [Security & Privacy Questions](#security--privacy-questions)
5. [Troubleshooting Quick Answers](#troubleshooting-quick-answers)

---

## General Questions

### Q1: What is AutoAttendance?

**A**: AutoAttendance is an intelligent face recognition-based attendance system for educational institutions. It combines deep learning models (YuNet for detection, ArcFace for recognition, Silent-Face for liveness) into a real-time pipeline that marks student attendance automatically based on facial recognition.

**Key features**:
- Real-time face detection and tracking
- Multi-frame confirmation voting (reduces false positives)
- Anti-spoofing defense (prevents fake face presentations)
- Session-based attendance model
- Web interface for admins and students

### Q2: Is AutoAttendance open-source?

**A**: Yes! The source code is available on GitHub: [github.com/ShubhamPatra/attendance_system](https://github.com/ShubhamPatra/attendance_system) under the MIT License (or as per your installation).

You can:
- Download and modify the code
- Deploy on your own servers
- Contribute improvements via pull requests

### Q3: What are the system requirements?

**A**: 

**Minimum (CPU-only)**:
- Python 3.9–3.13
- 4 GB RAM, 8 GB recommended
- 20 GB storage (includes models)

**Recommended (with GPU)**:
- NVIDIA RTX 2080 or better
- CUDA Toolkit 11.8+
- cuDNN 8.6+

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for complete setup instructions.

### Q4: How much does AutoAttendance cost?

**A**:

| Deployment | Cost |
|---|---|
| **Self-hosted (on-premises)** | Free (open-source) + infrastructure |
| **Cloud (AWS/GCP/Azure)** | ~$50–200/month depending on scale |
| **MongoDB Atlas (cloud)** | Free tier (< 512 MB) or $9+/month |

No licensing fees. See [DEPLOYMENT.md](DEPLOYMENT.md) for cost estimates.

---

## Technical Capability Questions

### Q5: What accuracy does the system achieve?

**A**: Under ideal conditions (controlled lighting, reasonable faces):
- **Recognition accuracy**: 99.1% (student correctly identified)
- **False positive rate**: 0.3% (wrong student marked)
- **False negative rate**: 0.6% (student not marked)
- **Anti-spoofing accuracy**: 97.8% (attacks detected)

Real-world accuracy varies by environment. See [RESEARCH.md](RESEARCH.md) for detailed metrics.

### Q6: Can the system work outdoors or in poor lighting?

**A**: Challenging but possible.

**Poor lighting**:
- Minimum: 50 lux (very dim) - 40% detection rate
- Recommended: 500+ lux - 99% detection rate
- **Workaround**: Use infrared illumination + IR-capable camera

**Outdoor**:
- Backlighting problems (student in shadow)
- **Workaround**: Adjust camera positioning or use anti-glare screens

See [LIMITATIONS.md](LIMITATIONS.md) for details.

### Q7: How many students can the system handle simultaneously?

**A**: Per camera: **10–15 students** for reliable recognition.

**Limitations**:
- Single camera limited by CSRT tracker latency
- 15+ faces → 5–10% ID swap rate (false positives)

**Solution**: Deploy multiple cameras.

**Database scale**:
- Single MongoDB: 100K+ students supported
- Sharded MongoDB: 1M+ students supported

### Q8: Can students use the system from their phones?

**A**: Currently: No direct mobile support.

**Workarounds**:
- Web portal accessible from phone browser (read-only enrollment status)
- QR code check-in (planned feature)

**Future**: Native iOS/Android mobile apps planned for Semester 2 (see [FUTURE.md](FUTURE.md)).

### Q9: What happens if the internet is down?

**A**: Depends on deployment:

| Deployment | Behavior |
|---|---|
| **Local MongoDB** | Continues (no internet needed) |
| **MongoDB Atlas** | Fails (cloud connection required) |
| **Air-gapped** | Fully offline operation (with custom sync) |

**Recommendation**: Keep MongoDB instance local or use hybrid mode (local cache + cloud sync).

### Q10: Can the system distinguish identical twins?

**A**: Not reliably (limitation of face recognition in general).

**Current accuracy**: 91.8% (vs. 99.1% for non-twins).

**Workarounds**:
- Supplement with student ID card verification
- Use multi-biometric (face + iris)
- Manual admin verification for twins

---

## Deployment Questions

### Q11: How do I deploy to production?

**A**: See [DEPLOYMENT.md](DEPLOYMENT.md) for step-by-step guide.

**Quick summary**:
```bash
# 1. Local setup
git clone <repo>
pip install -r requirements.txt
python run_admin.py  # Development

# 2. Docker deployment
docker-compose up -d  # CPU
INSTALL_GPU=1 docker-compose up -d  # GPU

# 3. Cloud deployment (AWS example)
# Instance: t3.xlarge (4 vCPU, 16 GB RAM)
# MongoDB: Atlas free tier
# Frontend: Nginx on EC2
# Cost: ~$50/month
```

### Q12: Can I use my existing MongoDB database?

**A**: Yes!

**Setup**:
```bash
# Set MongoDB URI in .env
MONGODB_URI=mongodb+srv://user:password@your-cluster.mongodb.net/
MONGODB_DATABASE=attendance_system

# Run bootstrap (creates collections & indexes)
python scripts/bootstrap_admin.py
```

### Q13: How do I integrate with my university's student database?

**A**: Bulk import via CSV:

```bash
# CSV format: registration_number, name, email, semester, course_ids
python scripts/bulk_import_students.py --file=students.csv
```

Or manually import via admin dashboard UI.

### Q14: Can I export attendance records?

**A**: Yes, multiple formats:

```bash
# Via API
curl http://localhost:5000/api/attendance/export \
  -d "{'date': '2024-09-15', 'format': 'csv'}" \
  -o attendance.csv

# Via CLI
mongoexport --uri="mongodb://..." --collection=attendance --out=data.json
```

---

## Security & Privacy Questions

### Q15: Where is biometric data stored?

**A**: 

| Data | Location | Encryption |
|---|---|---|
| **Face embeddings** | MongoDB | At-rest (AES-256) |
| **Student photos** | File storage | At-rest (AES-256) |
| **Attendance marks** | MongoDB | At-rest (AES-256) |

**Transit**: TLS 1.3 (encrypted).

### Q16: How long is biometric data kept?

**A**: **Default**: Deleted after 1 year (configurable).

**Policy**:
- Embeddings: 1 year (tunable)
- Raw face images: 30 days
- Attendance records: Indefinite (audit trail)

**GDPR compliance**:
- Students can request deletion
- Right to be forgotten implemented
- Data export available

### Q17: Can someone hack the system and gain unauthorized access?

**A**: 

**Vulnerabilities addressed**:
- ✓ Anti-spoofing defense (can't use photos/videos)
- ✓ HTTPS/TLS encryption
- ✓ Session-based auth (cookies)
- ✓ Role-based access control (RBAC)

**Remaining risks** (mitigated but not eliminated):
- Deep fake attacks (high-quality synthetic videos)
- Social engineering (admin credential compromise)
- Physical access to server

**Recommendations**:
- Use strong admin passwords
- Enable 2FA (when implemented)
- Regular security audits
- Keep system updated

See [LIMITATIONS.md](LIMITATIONS.md) for security considerations.

### Q18: Is the system compliant with GDPR/CCPA?

**A**: 

| Regulation | Status | Notes |
|---|---|---|
| **GDPR** | Partially | Consent mechanisms, data deletion pipeline implemented |
| **CCPA** | Partially | Right to access/deletion supported |
| **FERPA** | Yes | Student data protected (US) |

**Action required**: Consult legal team for institutional compliance.

---

## Troubleshooting Quick Answers

### Q19: System marks the wrong student as present

**A**: **Root cause**: Recognition threshold too low.

**Quick fix**:
```bash
export ATTENDANCE_RECOGNITION_THRESHOLD=0.42  # More strict
export ATTENDANCE_RECOGNITION_CONFIRM_FRAMES=3  # More voting
python run_admin.py  # Restart
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#high-false-positive-rate) for details.

### Q20: Student not recognized despite being enrolled

**A**: **Root cause**: Face changed or enrollment quality low.

**Solution**:
```bash
# 1. Re-enroll student with better quality samples
# Student navigates to self-enrollment portal and retakes photos

# 2. Or manually verify enrollment
mongosh
> use attendance_system
> db.students.findOne({ registration_number: 'CS21001' })
# Check if face_embedding present
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#high-false-negative-rate) for details.

### Q21: FPS is very low (< 10 FPS)

**A**: **Quick fixes** (in order):
1. Reduce frame width: `ATTENDANCE_FRAME_PROCESS_WIDTH=384`
2. Increase detection interval: `ATTENDANCE_DETECTION_INTERVAL=8`
3. Enable GPU: `ATTENDANCE_ENABLE_GPU=1`

See [OPTIMIZATION.md](OPTIMIZATION.md#latency-optimization) for detailed tuning.

### Q22: "MongoDB connection refused" error

**A**: **Causes**:
- MongoDB not running
- Wrong connection string
- Firewall blocking

**Fix**:
```bash
# Check if running
sudo systemctl status mongod

# Or for MongoDB Atlas
# Verify connection string in .env:
# MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#mongodb-connection-refused) for details.

### Q23: Models not found error on startup

**A**: **Fix**:
```bash
python scripts/download_models.py
# Takes 5–10 minutes
```

### Q24: Camera not opening / No video

**A**: **Causes**:
- Camera not connected
- Permission denied
- Wrong camera ID

**Fix**:
```bash
# List cameras
ls /dev/video*  # Linux
# Camera is video0, video1, etc.

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

---

## Need More Help?

- **Documentation**: See [docs/](../) directory
- **GitHub Issues**: Report bugs at [github.com/ShubhamPatra/attendance_system/issues](https://github.com/ShubhamPatra/attendance_system/issues)
- **Detailed Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Build Guide**: [BUILD_GUIDE.md](BUILD_GUIDE.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0
