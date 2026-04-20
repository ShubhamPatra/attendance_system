"""
Analytics and anomaly detection routes.

PHASE 5: Provides REST API for:
- Recent suspicious activity (spoof attempts, multi-identity detections)
- Per-student attendance trends
- Class-wise attendance statistics
- Dropout detection

Used by admin dashboard for security monitoring and analytics.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List

from flask import Blueprint, jsonify, request
import bson

import core.config as config
from core.utils import setup_logging
import core.database as database
from core.security_logs import get_anomaly_detector

logger = setup_logging()

analytics_bp = Blueprint("analytics", __name__, url_prefix="/api/analytics")


def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return None
    
    # Convert ObjectId to string
    if "_id" in doc and isinstance(doc["_id"], bson.ObjectId):
        doc["_id"] = str(doc["_id"])
    
    if "student_id" in doc and isinstance(doc["student_id"], bson.ObjectId):
        doc["student_id"] = str(doc["student_id"])
    
    # Convert datetime to ISO string
    for key in ["timestamp", "date", "recorded_at"]:
        if key in doc and isinstance(doc[key], datetime):
            doc[key] = doc[key].isoformat()
    
    return doc


@analytics_bp.route("/anomalies", methods=["GET"])
def get_anomalies():
    """Get recent security anomalies.
    
    Query parameters:
    - hours: How many hours to look back (default 24)
    - event_type: Filter by type (spoof_attempt, multi_identity, etc.)
    - severity: Filter by severity (low, medium, high, critical)
    - student_id: Filter by student ID
    - limit: Max results (default 100)
    
    Returns:
        {
            "success": bool,
            "anomalies": [event_docs],
            "count": int,
            "time_window": {start_time, end_time}
        }
    """
    try:
        hours = int(request.args.get("hours", 24))
        event_type = request.args.get("event_type")
        severity = request.args.get("severity")
        student_id_str = request.args.get("student_id")
        limit = int(request.args.get("limit", 100))

        db = database.get_db()
        
        # Build query
        query: Dict[str, Any] = {}
        
        # Time range
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        query["timestamp"] = {"$gte": cutoff_time}
        
        # Event type filter
        if event_type:
            query["type"] = event_type
        
        # Severity filter
        if severity:
            query["severity"] = severity
        
        # Student filter
        if student_id_str:
            try:
                query["student_id"] = bson.ObjectId(student_id_str)
            except:
                return jsonify({"success": False, "error": "Invalid student_id"}), 400
        
        # Query
        anomalies = list(
            db.security_logs
            .find(query)
            .sort("timestamp", -1)
            .limit(limit)
        )
        
        # Serialize
        anomalies = [_serialize_doc(doc) for doc in anomalies]
        
        return jsonify({
            "success": True,
            "anomalies": anomalies,
            "count": len(anomalies),
            "time_window": {
                "start": cutoff_time.isoformat(),
                "end": datetime.utcnow().isoformat(),
                "hours": hours,
            },
        }), 200

    except Exception as e:
        logger.error("Failed to get anomalies: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@analytics_bp.route("/attendance-trends", methods=["GET"])
def get_attendance_trends():
    """Get per-student attendance trends.
    
    Query parameters:
    - student_id: Filter by specific student (if provided)
    - days: How many days to look back (default 30)
    - limit: Max students to return (default 50)
    
    Returns:
        {
            "success": bool,
            "trends": [
                {
                    "student_id": str,
                    "student_name": str,
                    "total_present": int,
                    "total_absent": int,
                    "attendance_rate": float (0-1),
                    "recent_attendance": [date],
                    "dropout_detected": bool,
                    "last_present_date": str,
                    "consecutive_absences": int,
                }
            ],
            "time_window": {start_date, end_date, days}
        }
    """
    try:
        student_id_str = request.args.get("student_id")
        days = int(request.args.get("days", 30))
        limit = int(request.args.get("limit", 50))
        
        db = database.get_db()
        
        # Build query
        query: Dict[str, Any] = {}
        
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        query["date"] = {"$gte": cutoff_date}
        
        if student_id_str:
            try:
                query["student_id"] = bson.ObjectId(student_id_str)
            except:
                return jsonify({"success": False, "error": "Invalid student_id"}), 400
        
        # Aggregation pipeline
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": "$student_id",
                    "total_present": {"$sum": 1},
                    "last_present_date": {"$max": "$date"},
                    "dates": {"$push": "$date"},
                    "last_time": {"$max": "$time"},
                }
            },
            {"$sort": {"total_present": -1}},
            {"$limit": limit},
        ]
        
        results = list(db.attendance.aggregate(pipeline))
        
        # Enrich with computed fields
        trends = []
        for result in results:
            student_id_obj = result["_id"]
            student = database.get_student_by_id(student_id_obj)
            
            # Get all class dates in range
            all_class_dates = set()
            try:
                class_docs = list(db.classes.find({}, {"date": 1}))
                for class_doc in class_docs:
                    if class_doc.get("date", "") >= cutoff_date:
                        all_class_dates.add(class_doc.get("date", ""))
            except:
                pass
            
            present_dates = set(result.get("dates", []))
            absent_dates = all_class_dates - present_dates
            
            # Consecutive absence check
            consecutive_absences = 0
            sorted_dates = sorted(all_class_dates, reverse=True)
            for date in sorted_dates:
                if date not in present_dates:
                    consecutive_absences += 1
                else:
                    break
            
            attendance_rate = 0.0
            if all_class_dates:
                attendance_rate = len(present_dates) / len(all_class_dates)
            
            anomaly_detector = get_anomaly_detector()
            dropout = anomaly_detector.detect_dropout(student_id_obj, absent_day_threshold=3)
            
            trends.append({
                "student_id": str(student_id_obj),
                "student_name": student.get("name", "") if student else "Unknown",
                "total_present": result.get("total_present", 0),
                "total_absent": len(absent_dates),
                "attendance_rate": round(attendance_rate, 3),
                "recent_attendance": sorted(present_dates)[-7:],  # Last 7 days
                "dropout_detected": dropout,
                "last_present_date": result.get("last_present_date"),
                "consecutive_absences": consecutive_absences,
            })
        
        return jsonify({
            "success": True,
            "trends": trends,
            "count": len(trends),
            "time_window": {
                "start_date": cutoff_date,
                "end_date": datetime.utcnow().date().isoformat(),
                "days": days,
            },
        }), 200

    except Exception as e:
        logger.error("Failed to get attendance trends: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@analytics_bp.route("/class-stats", methods=["GET"])
def get_class_stats():
    """Get class-wise attendance statistics.
    
    Query parameters:
    - class_id: Filter by specific class (if provided)
    - days: How many days to look back (default 30)
    - limit: Max classes to return (default 50)
    
    Returns:
        {
            "success": bool,
            "stats": [
                {
                    "class_id": str,
                    "class_name": str,
                    "total_students": int,
                    "total_present": int,
                    "avg_attendance_rate": float (0-1),
                    "peak_time": str (HH:MM),
                    "attendance_by_hour": {hour: count},
                    "high_dropout_students": int,
                }
            ],
            "time_window": {start_date, end_date, days}
        }
    """
    try:
        class_id_str = request.args.get("class_id")
        days = int(request.args.get("days", 30))
        limit = int(request.args.get("limit", 50))
        
        db = database.get_db()
        
        # Build query
        query: Dict[str, Any] = {}
        
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        query["date"] = {"$gte": cutoff_date}
        
        if class_id_str:
            try:
                query["class_id"] = bson.ObjectId(class_id_str)
            except:
                return jsonify({"success": False, "error": "Invalid class_id"}), 400
        
        # Aggregation pipeline
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": "$class_id",
                    "total_present": {"$sum": 1},
                    "unique_students": {"$addToSet": "$student_id"},
                    "times": {"$push": "$time"},
                }
            },
            {"$sort": {"total_present": -1}},
            {"$limit": limit},
        ]
        
        results = list(db.attendance.aggregate(pipeline))
        
        # Enrich with computed fields
        stats = []
        for result in results:
            class_id_obj = result["_id"]
            class_doc = database.get_class_by_id(class_id_obj) if hasattr(database, "get_class_by_id") else None
            
            unique_students = result.get("unique_students", [])
            attendance_rate = 0.0
            if len(unique_students) > 0:
                attendance_rate = result.get("total_present", 0) / len(unique_students)
            
            # Peak time computation
            peak_time = "00:00"
            times = result.get("times", [])
            if times:
                # Extract hours from times
                hour_counts = {}
                for time_str in times:
                    try:
                        hour = int(time_str.split(":")[0])
                        hour_counts[hour] = hour_counts.get(hour, 0) + 1
                    except:
                        pass
                
                if hour_counts:
                    peak_hour = max(hour_counts, key=hour_counts.get)
                    peak_time = f"{peak_hour:02d}:00"
            
            # Dropout detection for students in this class
            high_dropout_students = 0
            anomaly_detector = get_anomaly_detector()
            for student_id in unique_students:
                if anomaly_detector.detect_dropout(student_id, absent_day_threshold=3):
                    high_dropout_students += 1
            
            stats.append({
                "class_id": str(class_id_obj),
                "class_name": class_doc.get("name", "") if class_doc else "Unknown",
                "total_students": len(unique_students),
                "total_present": result.get("total_present", 0),
                "avg_attendance_rate": round(attendance_rate, 3),
                "peak_time": peak_time,
                "high_dropout_students": high_dropout_students,
            })
        
        return jsonify({
            "success": True,
            "stats": stats,
            "count": len(stats),
            "time_window": {
                "start_date": cutoff_date,
                "end_date": datetime.utcnow().date().isoformat(),
                "days": days,
            },
        }), 200

    except Exception as e:
        logger.error("Failed to get class stats: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


def register_analytics_routes(app):
    """Register analytics blueprint with Flask app.
    
    Call this in app.py:
        from web.anomaly_routes import register_analytics_routes
        register_analytics_routes(app)
    """
    app.register_blueprint(analytics_bp)
