"""
Analytics API routes for attendance dashboard.

Provides REST endpoints for analytics data with Plotly-compatible JSON responses.
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import datetime

import core.config as config
import core.analytics_pipelines as analytics
from core.utils import setup_logging, sanitize_string

logger = setup_logging()

bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')


def require_admin(f):
    """Decorator to require admin authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # TODO: Add auth check here (use existing auth system)
        # For now, allow all requests
        return f(*args, **kwargs)
    return decorated_function


def _validate_days_param(days_str: str, default: int = 30) -> int:
    """
    Validate and sanitize days parameter.
    
    Args:
        days_str: String from request param
        default: Default value if invalid
        
    Returns:
        Validated days (1-365)
    """
    try:
        days = int(days_str)
        days = max(1, min(365, days))
        return days
    except (ValueError, TypeError):
        return default


@bp.route('/overview', methods=['GET'])
@require_admin
def get_overview():
    """
    GET /api/analytics/overview?days=7
    
    Get overview KPI metrics for dashboard.
    
    Query Parameters:
        days: Number of days (1-365, default 7)
    
    Returns:
        {
            "success": true,
            "data": {
                "total_students": int,
                "present_today": int,
                "late_count": int,
                "absent_count": int,
                "avg_attendance_percent": float,
                "on_time_percent": float
            }
        }
    """
    try:
        days = _validate_days_param(request.args.get('days', '7'))
        
        data = analytics.get_analytics_overview(days=days)
        
        return jsonify({
            "success": True,
            "data": data,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }), 200
    
    except Exception as exc:
        logger.error(f"Overview endpoint error: {exc}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch overview data"
        }), 500


@bp.route('/attendance-trend', methods=['GET'])
@require_admin
def get_attendance_trend():
    """
    GET /api/analytics/attendance-trend?days=30
    
    Get daily attendance trend data for line chart.
    
    Query Parameters:
        days: Number of days (1-365, default 30)
    
    Returns:
        {
            "success": true,
            "data": [
                {
                    "date": "2025-04-16",
                    "total": 100,
                    "present": 92,
                    "late": 15,
                    "absent": 8,
                    "present_percent": 92.0
                },
                ...
            ]
        }
    """
    try:
        days = _validate_days_param(request.args.get('days', '30'))
        
        data = analytics.get_attendance_trend_daily(days=days)
        
        return jsonify({
            "success": True,
            "data": data,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }), 200
    
    except Exception as exc:
        logger.error(f"Attendance trend endpoint error: {exc}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch attendance trend"
        }), 500


@bp.route('/late-stats', methods=['GET'])
@require_admin
def get_late_stats():
    """
    GET /api/analytics/late-stats?days=30
    
    Get late arrival statistics.
    
    Query Parameters:
        days: Number of days (1-365, default 30)
    
    Returns:
        {
            "success": true,
            "data": {
                "daily_trend": [...],
                "peak_late_hour": 9,
                "total_late": 142,
                "top_late_students": [...]
            }
        }
    """
    try:
        days = _validate_days_param(request.args.get('days', '30'))
        
        data = analytics.get_late_statistics(days=days)
        
        return jsonify({
            "success": True,
            "data": data,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }), 200
    
    except Exception as exc:
        logger.error(f"Late stats endpoint error: {exc}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch late statistics"
        }), 500


@bp.route('/heatmap-v2', methods=['GET'])
@require_admin
def get_heatmap():
    """
    GET /api/analytics/heatmap-v2?days=90
    
    Get enhanced heatmap data with confidence scores.
    
    Query Parameters:
        days: Number of days (1-365, default 90)
    
    Returns:
        {
            "success": true,
            "data": [
                {
                    "date": "2025-01-15",
                    "total_present": 85,
                    "total_students": 100,
                    "attendance_percent": 85.0,
                    "avg_confidence": 0.9234
                },
                ...
            ]
        }
    """
    try:
        days = _validate_days_param(request.args.get('days', '90'))
        
        data = analytics.get_heatmap_enhanced(days=days)
        
        return jsonify({
            "success": True,
            "data": data,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }), 200
    
    except Exception as exc:
        logger.error(f"Heatmap endpoint error: {exc}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch heatmap data"
        }), 500


@bp.route('/course-breakdown', methods=['GET'])
@require_admin
def get_course_breakdown():
    """
    GET /api/analytics/course-breakdown?days=30&course_id=optional
    
    Get attendance breakdown by course/section.
    
    Query Parameters:
        days: Number of days (1-365, default 30)
        course_id: Optional specific course filter
    
    Returns:
        {
            "success": true,
            "data": [
                {
                    "course": "CS-101",
                    "section": "A",
                    "total_students": 40,
                    "present": 38,
                    "absent": 2,
                    "attendance_percent": 95.0
                },
                ...
            ]
        }
    """
    try:
        days = _validate_days_param(request.args.get('days', '30'))
        course_id = request.args.get('course_id', None)
        if course_id:
            course_id = sanitize_string(course_id)
        
        data = analytics.get_course_attendance_breakdown(
            course_id=course_id,
            days=days
        )
        
        return jsonify({
            "success": True,
            "data": data,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }), 200
    
    except Exception as exc:
        logger.error(f"Course breakdown endpoint error: {exc}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch course breakdown"
        }), 500


@bp.errorhandler(400)
def handle_bad_request(e):
    """Handle 400 errors."""
    return jsonify({
        "success": False,
        "error": "Bad request"
    }), 400


@bp.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@bp.errorhandler(500)
def handle_internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


def register_analytics_routes(app_bp):
    """Register analytics routes with the main blueprint.
    
    Parameters
    ----------
    app_bp : Flask.Blueprint
        The main application blueprint to register routes with
    """
    app_bp.register_blueprint(bp)
