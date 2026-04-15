from flask_wtf import FlaskForm
from wtforms import BooleanField, IntegerField, PasswordField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, Length


class LoginForm(FlaskForm):
	email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
	password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=128)])
	remember_me = BooleanField("Remember me")
	submit = SubmitField("Sign In")


class ChangePasswordForm(FlaskForm):
	old_password = PasswordField("Current Password", validators=[DataRequired(), Length(min=8, max=128)])
	new_password = PasswordField("New Password", validators=[DataRequired(), Length(min=8, max=128)])
	submit = SubmitField("Update Password")


class StudentForm(FlaskForm):
	student_id = StringField("Student ID", validators=[DataRequired(), Length(min=3, max=30)])
	name = StringField("Name", validators=[DataRequired(), Length(min=2, max=120)])
	email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
	department = StringField("Department", validators=[DataRequired(), Length(max=50)])
	year = IntegerField("Year", validators=[DataRequired()])
	face_photo_path = StringField("Face Photo Path", validators=[Length(max=255)])
	submit = SubmitField("Save Student")


class CourseForm(FlaskForm):
	course_code = StringField("Course Code", validators=[DataRequired(), Length(min=2, max=30)])
	course_name = StringField("Course Name", validators=[DataRequired(), Length(min=2, max=120)])
	department = StringField("Department", validators=[DataRequired(), Length(max=50)])
	instructor = StringField("Instructor", validators=[DataRequired(), Length(max=120)])
	schedule_json = TextAreaField("Schedule JSON", validators=[DataRequired()])
	submit = SubmitField("Save Course")


class AttendanceForm(FlaskForm):
	student_id = StringField("Student ID", validators=[DataRequired(), Length(min=3, max=30)])
	course_id = StringField("Course ID", validators=[DataRequired(), Length(min=3, max=40)])
	status = StringField("Status", validators=[DataRequired(), Length(min=3, max=20)])
	reason = StringField("Reason", validators=[Length(max=255)])
	submit = SubmitField("Save Attendance")


class ReportForm(FlaskForm):
	report_type = StringField("Report Type", validators=[DataRequired(), Length(min=3, max=30)])
	date_from = StringField("Date From", validators=[DataRequired(), Length(min=10, max=10)])
	date_to = StringField("Date To", validators=[DataRequired(), Length(min=10, max=10)])
	course_id = StringField("Course ID", validators=[Length(max=40)])
	student_id = StringField("Student ID", validators=[Length(max=40)])
	submit = SubmitField("Generate Report")
