"""
Analytics pipelines module for MongoDB aggregation queries.

Provides comprehensive analytics for attendance dashboard including trends,
late statistics, heatmaps, and per-course breakdowns.
"""

import datetime
from typing import List, Dict, Optional

import core.config as config
import core.database as database
from core.utils import setup_logging

logger = setup_logging()


def get_analytics_overview(days: int = 7) -> Dict:
    """
    Get overview KPI metrics for attendance dashboard.
    
    Returns metrics for a given time window.
    
    Args:
        days: Number of days to include in overview (1-365)
        
    Returns:
        Dict with keys:
        - total_students: int
        - present_today: int
        - late_count: int (today only)
        - absent_count: int (today only)
        - avg_attendance_percent: float
        - on_time_percent: float (percentage of on-time arrivals)
    """
    try:
        # Validate days parameter
        days = max(1, min(365, int(days)))
        
        # Calculate date range
        today = datetime.datetime.now().date()
        start_date = str(today - datetime.timedelta(days=days - 1))
        end_date = str(today)
        
        # Get database connection
        db = database.get_db()
        
        # Total students
        total_students = db.students.count_documents({"is_active": True})
        
        # Present today
        today_str = str(today)
        present_today = db.attendance.count_documents({
            "date": today_str,
            "status": "Present"
        })
        
        # Late count today (after cutoff time)
        cutoff_time = config.LATE_ARRIVAL_CUTOFF_TIME  # "09:00:00"
        late_pipeline = [
            {
                "$match": {
                    "date": today_str,
                    "status": "Present"
                }
            },
            {
                "$project": {
                    "time_str": "$time",
                    "is_late": {
                        "$gte": [
                            {
                                "$substr": ["$time", 0, 2]
                            },
                            {
                                "$substr": [cutoff_time, 0, 2]
                            }
                        ]
                    }
                }
            },
            {
                "$match": {"is_late": True}
            },
            {
                "$count": "count"
            }
        ]
        late_result = list(db.attendance.aggregate(late_pipeline))
        late_count = late_result[0]["count"] if late_result else 0
        
        # Absent count today
        absent_count = total_students - present_today
        
        # Average attendance percentage over date range
        attendance_pipeline = [
            {
                "$match": {
                    "date": {"$gte": start_date, "$lte": end_date},
                    "status": "Present"
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "count": {"$sum": 1}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_count": {"$avg": "$count"}
                }
            }
        ]
        avg_result = list(db.attendance.aggregate(attendance_pipeline))
        avg_present = avg_result[0]["avg_count"] if avg_result else 0
        avg_attendance_percent = (avg_present / max(total_students, 1)) * 100.0
        
        # On-time percentage today
        on_time_today = present_today - late_count
        on_time_percent = (on_time_today / max(present_today, 1)) * 100.0 if present_today > 0 else 0.0
        
        return {
            "total_students": total_students,
            "present_today": present_today,
            "late_count": late_count,
            "absent_count": absent_count,
            "avg_attendance_percent": round(avg_attendance_percent, 2),
            "on_time_percent": round(on_time_percent, 2),
        }
        
    except Exception as exc:
        logger.error(f"get_analytics_overview failed: {exc}")
        return {
            "total_students": 0,
            "present_today": 0,
            "late_count": 0,
            "absent_count": 0,
            "avg_attendance_percent": 0.0,
            "on_time_percent": 0.0,
        }


def get_attendance_trend_daily(days: int = 30) -> List[Dict]:
    """
    Get daily attendance trend data.
    
    Args:
        days: Number of days to retrieve (1-365)
        
    Returns:
        List of dicts with keys: date, total, present, late, absent, present_percent
    """
    try:
        days = max(1, min(365, int(days)))
        
        db = database.get_db()
        today = datetime.datetime.now().date()
        start_date = str(today - datetime.timedelta(days=days - 1))
        
        # Get all attendance records for date range
        pipeline = [
            {
                "$match": {
                    "date": {"$gte": start_date},
                    "status": "Present"
                }
            },
            {
                "$project": {
                    "date": 1,
                    "time": 1
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "count": {"$sum": 1},
                    "times": {"$push": "$time"}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        results = list(db.attendance.aggregate(pipeline))
        
        # Get total students
        total_students = db.students.count_documents({"is_active": True})
        
        # Build response with late counts
        cutoff_time = config.LATE_ARRIVAL_CUTOFF_TIME
        trend_data = []
        
        for record in results:
            date = record["_id"]
            present = record["count"]
            times = record.get("times", [])
            
            # Count late arrivals for this date
            late_count = sum(1 for t in times if t >= cutoff_time)
            
            # Calculate percentages
            present_percent = (present / max(total_students, 1)) * 100.0 if total_students > 0 else 0.0
            
            trend_data.append({
                "date": date,
                "total": total_students,
                "present": present,
                "late": late_count,
                "absent": total_students - present,
                "present_percent": round(present_percent, 2),
            })
        
        return trend_data
        
    except Exception as exc:
        logger.error(f"get_attendance_trend_daily failed: {exc}")
        return []


def get_late_statistics(days: int = 30) -> Dict:
    """
    Get late arrival statistics.
    
    Args:
        days: Number of days to analyze (1-365)
        
    Returns:
        Dict with:
        - daily_trend: list of {date, late_count, on_time_count}
        - peak_late_hour: hour with most late arrivals
        - total_late: total late arrivals in period
        - top_late_students: list of {name, reg_no, late_count}
    """
    try:
        days = max(1, min(365, int(days)))
        
        db = database.get_db()
        today = datetime.datetime.now().date()
        start_date = str(today - datetime.timedelta(days=days - 1))
        cutoff_time = config.LATE_ARRIVAL_CUTOFF_TIME
        
        # Get daily late counts
        daily_pipeline = [
            {
                "$match": {
                    "date": {"$gte": start_date},
                    "status": "Present"
                }
            },
            {
                "$project": {
                    "date": 1,
                    "time": 1,
                    "student_id": 1,
                    "is_late": {"$gte": ["$time", cutoff_time]}
                }
            },
            {
                "$group": {
                    "_id": {
                        "date": "$date",
                        "is_late": "$is_late"
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id.date": 1}
            }
        ]
        
        daily_results = list(db.attendance.aggregate(daily_pipeline))
        
        # Process daily results
        daily_trend = {}
        for record in daily_results:
            date = record["_id"]["date"]
            is_late = record["_id"]["is_late"]
            count = record["count"]
            
            if date not in daily_trend:
                daily_trend[date] = {"late_count": 0, "on_time_count": 0}
            
            if is_late:
                daily_trend[date]["late_count"] = count
            else:
                daily_trend[date]["on_time_count"] = count
        
        # Convert to list
        daily_list = [
            {
                "date": date,
                "late_count": data["late_count"],
                "on_time_count": data["on_time_count"]
            }
            for date, data in sorted(daily_trend.items())
        ]
        
        # Get peak late hour
        hour_pipeline = [
            {
                "$match": {
                    "date": {"$gte": start_date},
                    "status": "Present",
                    "time": {"$gte": cutoff_time}
                }
            },
            {
                "$project": {
                    "hour": {"$substr": ["$time", 0, 2]}
                }
            },
            {
                "$group": {
                    "_id": "$hour",
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            },
            {
                "$limit": 1
            }
        ]
        
        hour_results = list(db.attendance.aggregate(hour_pipeline))
        peak_late_hour = int(hour_results[0]["_id"]) if hour_results else 9
        
        # Get top late students
        top_students_pipeline = [
            {
                "$match": {
                    "date": {"$gte": start_date},
                    "status": "Present",
                    "time": {"$gte": cutoff_time}
                }
            },
            {
                "$group": {
                    "_id": "$student_id",
                    "late_count": {"$sum": 1}
                }
            },
            {
                "$sort": {"late_count": -1}
            },
            {
                "$limit": 10
            },
            {
                "$lookup": {
                    "from": "students",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "student_info"
                }
            },
            {
                "$unwind": "$student_info"
            },
            {
                "$project": {
                    "name": "$student_info.name",
                    "reg_no": "$student_info.registration_number",
                    "late_count": 1
                }
            }
        ]
        
        top_students = list(db.attendance.aggregate(top_students_pipeline))
        
        # Total late count
        total_late = sum(d["late_count"] for d in daily_list)
        
        return {
            "daily_trend": daily_list,
            "peak_late_hour": peak_late_hour,
            "total_late": total_late,
            "top_late_students": top_students,
        }
        
    except Exception as exc:
        logger.error(f"get_late_statistics failed: {exc}")
        return {
            "daily_trend": [],
            "peak_late_hour": 9,
            "total_late": 0,
            "top_late_students": [],
        }


def get_heatmap_enhanced(days: int = 90) -> List[Dict]:
    """
    Get enhanced heatmap data with confidence scores.
    
    Args:
        days: Number of days (1-365)
        
    Returns:
        List of dicts with: date, total_present, total_students, attendance_percent, avg_confidence
    """
    try:
        days = max(1, min(365, int(days)))
        
        db = database.get_db()
        today = datetime.datetime.now().date()
        start_date = str(today - datetime.timedelta(days=days - 1))
        
        pipeline = [
            {
                "$match": {
                    "date": {"$gte": start_date},
                    "status": "Present"
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "present_count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence_score"}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        results = list(db.attendance.aggregate(pipeline))
        total_students = db.students.count_documents({"is_active": True})
        
        heatmap_data = []
        for record in results:
            date = record["_id"]
            present = record["present_count"]
            avg_conf = record.get("avg_confidence", 0.0)
            
            percent = (present / max(total_students, 1)) * 100.0
            
            heatmap_data.append({
                "date": date,
                "total_present": present,
                "total_students": total_students,
                "attendance_percent": round(percent, 2),
                "avg_confidence": round(float(avg_conf), 4),
            })
        
        return heatmap_data
        
    except Exception as exc:
        logger.error(f"get_heatmap_enhanced failed: {exc}")
        return []


def get_course_attendance_breakdown(
    course_id: Optional[str] = None,
    days: int = 30
) -> List[Dict]:
    """
    Get attendance breakdown by course/section.
    
    Args:
        course_id: Optional specific course to filter by
        days: Number of days to include
        
    Returns:
        List of dicts with: course, section, total_students, present, absent, attendance_percent
    """
    try:
        days = max(1, min(365, int(days)))
        
        db = database.get_db()
        today = datetime.datetime.now().date()
        start_date = str(today - datetime.timedelta(days=days - 1))
        
        # Get all active students grouped by section
        students = list(db.students.find(
            {"is_active": True},
            {"_id": 1, "name": 1, "section": 1}
        ))
        
        # Count attendance per section
        attendance_pipeline = [
            {
                "$match": {
                    "date": {"$gte": start_date},
                    "status": "Present"
                }
            },
            {
                "$lookup": {
                    "from": "students",
                    "localField": "student_id",
                    "foreignField": "_id",
                    "as": "student_info"
                }
            },
            {
                "$unwind": "$student_info"
            },
            {
                "$group": {
                    "_id": "$student_info.section",
                    "present_count": {"$sum": 1}
                }
            }
        ]
        
        attendance_by_section = list(db.attendance.aggregate(attendance_pipeline))
        attendance_map = {a["_id"]: a["present_count"] for a in attendance_by_section}
        
        # Build section breakdown
        section_breakdown = {}
        for student in students:
            section = student.get("section", "Unknown")
            if section not in section_breakdown:
                section_breakdown[section] = {"total": 0, "present": 0}
            section_breakdown[section]["total"] += 1
            section_breakdown[section]["present"] += attendance_map.get(section, 0)
        
        # Convert to list
        result = []
        for section, counts in sorted(section_breakdown.items()):
            total = counts["total"]
            present = counts["present"]
            percent = (present / max(total, 1)) * 100.0
            
            result.append({
                "course": section,
                "section": section,
                "total_students": total,
                "present": present,
                "absent": total - present,
                "attendance_percent": round(percent, 2),
            })
        
        return result
        
    except Exception as exc:
        logger.error(f"get_course_attendance_breakdown failed: {exc}")
        return []
