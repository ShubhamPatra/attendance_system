from __future__ import annotations

import csv

import mongomock

import tasks.report_tasks as report_tasks


def _seed_data(db):
	cs = db.courses.insert_one(
		{
			"course_code": "CS401",
			"course_name": "Distributed Systems",
			"department": "CS",
		}
	).inserted_id
	ee = db.courses.insert_one(
		{
			"course_code": "EE201",
			"course_name": "Signals",
			"department": "EE",
		}
	).inserted_id

	s1 = db.students.insert_one(
		{
			"student_id": "CS2026001",
			"name": "Alice",
			"department": "CS",
		}
	).inserted_id
	s2 = db.students.insert_one(
		{
			"student_id": "EE2026002",
			"name": "Bob",
			"department": "EE",
		}
	).inserted_id

	db.attendance_records.insert_many(
		[
			{
				"date": "2026-04-10",
				"course_id": cs,
				"student_id": s1,
				"status": "present",
				"confidence_score": 0.95,
				"anti_spoofing_score": 0.93,
			},
			{
				"date": "2026-04-11",
				"course_id": cs,
				"student_id": s1,
				"status": "late",
				"confidence_score": 0.88,
				"anti_spoofing_score": 0.9,
			},
			{
				"date": "2026-04-11",
				"course_id": ee,
				"student_id": s2,
				"status": "absent",
				"confidence_score": 0.5,
				"anti_spoofing_score": 0.6,
			},
		]
	)


def test_generate_course_report(monkeypatch, tmp_path):
	db = mongomock.MongoClient()["report_tasks"]
	_seed_data(db)
	monkeypatch.setattr("tasks.report_tasks._get_db", lambda: db)

	output = tmp_path / "course.csv"
	result = report_tasks.generate_report.run(
		"course",
		{"date_from": "2026-04-01", "date_to": "2026-04-30", "course_id": "", "student_id": ""},
		str(output),
	)

	assert result["status"] == "success"
	with output.open("r", encoding="utf-8", newline="") as handle:
		rows = list(csv.DictReader(handle))
	assert len(rows) == 3
	assert rows[0]["course_code"] in {"CS401", "EE201"}


def test_generate_student_report(monkeypatch, tmp_path):
	db = mongomock.MongoClient()["report_tasks"]
	_seed_data(db)
	monkeypatch.setattr("tasks.report_tasks._get_db", lambda: db)

	output = tmp_path / "student.csv"
	result = report_tasks.generate_report.run(
		"student",
		{"date_from": "2026-04-01", "date_to": "2026-04-30", "course_id": "", "student_id": ""},
		str(output),
	)

	assert result["status"] == "success"
	with output.open("r", encoding="utf-8", newline="") as handle:
		rows = list(csv.DictReader(handle))
	assert len(rows) >= 2
	row = next(r for r in rows if r["student_id"] == "CS2026001" and r["course_code"] == "CS401")
	assert row["present"] == "1"
	assert row["late"] == "1"


def test_generate_department_report(monkeypatch, tmp_path):
	db = mongomock.MongoClient()["report_tasks"]
	_seed_data(db)
	monkeypatch.setattr("tasks.report_tasks._get_db", lambda: db)

	output = tmp_path / "department.csv"
	result = report_tasks.generate_report.run(
		"department",
		{"date_from": "2026-04-01", "date_to": "2026-04-30", "course_id": "", "student_id": ""},
		str(output),
	)

	assert result["status"] == "success"
	with output.open("r", encoding="utf-8", newline="") as handle:
		rows = list(csv.DictReader(handle))
	assert any(r["department"] == "CS" for r in rows)
	assert any(r["department"] == "EE" for r in rows)