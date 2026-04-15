from flask_wtf import FlaskForm
from wtforms import IntegerField, PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length, NumberRange


class StudentLoginForm(FlaskForm):
	student_id = StringField("Student ID", validators=[DataRequired(), Length(min=3, max=30)])
	password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=128)])
	submit = SubmitField("Sign In")


class RegistrationForm(FlaskForm):
	name = StringField("Full Name", validators=[DataRequired(), Length(min=2, max=120)])
	student_id = StringField("Student ID", validators=[DataRequired(), Length(min=3, max=30)])
	email = StringField("Email", validators=[DataRequired(), Email(), Length(max=255)])
	password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=128)])
	confirm_password = PasswordField(
		"Confirm Password",
		validators=[DataRequired(), EqualTo("password", message="Passwords must match")],
	)
	department = StringField("Department", validators=[DataRequired(), Length(min=2, max=80)])
	year = IntegerField("Year", validators=[DataRequired(), NumberRange(min=1, max=8)])
	face_photo_path = StringField("Face Photo Path", validators=[Length(max=255)])
	submit = SubmitField("Create Account")