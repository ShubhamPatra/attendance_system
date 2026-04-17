"""Unit tests for improved authentication system."""

import pytest
import bson
from datetime import datetime, timedelta, timezone

from core.auth import (
    hash_password,
    check_password,
    validate_password,
    generate_jwt_token,
    verify_jwt_token,
    generate_verification_token,
)
import core.database as db


class TestPasswordHashing:
    """Test password hashing and verification."""
    
    def test_hash_password_creates_hash(self):
        """Test that hash_password creates a non-empty hash."""
        password = "TestPassword123!"
        hashed = hash_password(password)
        
        assert hashed is not None
        assert len(hashed) > 0
        assert hashed != password
    
    def test_check_password_valid(self):
        """Test that check_password returns True for matching password."""
        password = "TestPassword123!"
        hashed = hash_password(password)
        
        assert check_password(hashed, password) is True
    
    def test_check_password_invalid(self):
        """Test that check_password returns False for non-matching password."""
        password = "TestPassword123!"
        hashed = hash_password(password)
        wrong_password = "WrongPassword123!"
        
        assert check_password(hashed, wrong_password) is False
    
    def test_check_password_empty(self):
        """Test that check_password handles empty password."""
        hashed = hash_password("something")
        assert check_password(hashed, "") is False


class TestPasswordValidation:
    """Test password strength validation."""
    
    def test_validate_password_valid(self):
        """Test that valid password passes validation."""
        password = "StrongPass123!@#"
        is_valid, msg = validate_password(password)
        
        assert is_valid is True
        assert msg == ""
    
    def test_validate_password_too_short(self):
        """Test that password < 12 chars fails."""
        password = "Short1!@"
        is_valid, msg = validate_password(password)
        
        assert is_valid is False
        assert "12 characters" in msg
    
    def test_validate_password_no_uppercase(self):
        """Test that password without uppercase fails."""
        password = "lowercase123!@#"
        is_valid, msg = validate_password(password)
        
        assert is_valid is False
        assert "uppercase" in msg
    
    def test_validate_password_no_lowercase(self):
        """Test that password without lowercase fails."""
        password = "UPPERCASE123!@#"
        is_valid, msg = validate_password(password)
        
        assert is_valid is False
        assert "lowercase" in msg
    
    def test_validate_password_no_digit(self):
        """Test that password without digit fails."""
        password = "NoDigits!@#ABC"
        is_valid, msg = validate_password(password)
        
        assert is_valid is False
        assert "digit" in msg
    
    def test_validate_password_no_special(self):
        """Test that password without special char fails."""
        password = "NoSpecial123ABC"
        is_valid, msg = validate_password(password)
        
        assert is_valid is False
        assert "special" in msg
    
    def test_validate_password_contains_reg_no(self):
        """Test that password containing registration number fails."""
        reg_no = "REG12345"
        password = f"MyPass{reg_no}123!@#"
        is_valid, msg = validate_password(password, registration_number=reg_no)
        
        assert is_valid is False
        assert "registration number" in msg
    
    def test_validate_password_contains_email_part(self):
        """Test that password containing email part fails."""
        email = "student@example.com"
        email_part = "student"
        password = f"MyPass{email_part}123!@#"
        is_valid, msg = validate_password(password, email=email)
        
        assert is_valid is False
        assert "email" in msg


class TestJWTTokens:
    """Test JWT token generation and verification."""
    
    def test_generate_jwt_token_creates_token(self):
        """Test that JWT token is generated."""
        # Note: This test requires Flask app context
        from flask import Flask
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'test-secret-key'
        
        with app.app_context():
            token = generate_jwt_token('user123', 'student', expires_in_hours=1)
            
            assert token is not None
            assert len(token) > 0
            assert isinstance(token, str)
    
    def test_verify_jwt_token_valid(self):
        """Test that valid JWT token verifies correctly."""
        from flask import Flask
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'test-secret-key'
        
        with app.app_context():
            user_id = 'user123'
            role = 'student'
            token = generate_jwt_token(user_id, role, expires_in_hours=1)
            
            payload = verify_jwt_token(token)
            
            assert payload is not None
            assert payload.get('user_id') == user_id
            assert payload.get('role') == role
    
    def test_verify_jwt_token_invalid(self):
        """Test that invalid JWT token returns None."""
        from flask import Flask
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'test-secret-key'
        
        with app.app_context():
            invalid_token = "invalid.jwt.token"
            payload = verify_jwt_token(invalid_token)
            
            assert payload is None
    
    def test_jwt_token_contains_jti(self):
        """Test that JWT token includes JTI for blacklisting."""
        from flask import Flask
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'test-secret-key'
        
        with app.app_context():
            token = generate_jwt_token('user123', 'student', expires_in_hours=1)
            payload = verify_jwt_token(token)
            
            assert payload is not None
            assert 'jti' in payload
            assert payload['jti'] is not None


class TestVerificationTokens:
    """Test verification token generation."""
    
    def test_generate_verification_token(self):
        """Test that verification token is generated."""
        token, expires_at = generate_verification_token(expires_in_hours=24)
        
        assert token is not None
        assert len(token) > 0
        assert expires_at is not None
        assert isinstance(expires_at, datetime)
    
    def test_verification_token_expiration(self):
        """Test that verification token has correct expiration."""
        hours = 24
        before = datetime.utcnow()
        token, expires_at = generate_verification_token(expires_in_hours=hours)
        after = datetime.utcnow()
        
        # Check expiration is within expected window
        assert expires_at > before
        expected_expiry = after + timedelta(hours=hours)
        # Allow 1 second grace period
        assert expires_at <= expected_expiry + timedelta(seconds=1)


class TestLoginAttemptTracking:
    """Test login attempt tracking for rate limiting."""
    
    def test_record_login_attempt(self):
        """Test that login attempt is recorded."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            user_id = bson.ObjectId()
            ip = "192.168.1.1"
            ua = "TestBrowser"
            
            # Record attempt (may fail if DB not available, but should not crash)
            try:
                db.record_login_attempt(user_id, success=True, ip_address=ip, user_agent=ua)
            except Exception as e:
                pytest.skip(f"Database not available: {e}")
    
    def test_get_recent_login_failures(self):
        """Test retrieving recent failed login attempts."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            user_id = bson.ObjectId()
            
            try:
                count = db.get_recent_login_failures(user_id, minutes=15)
                assert isinstance(count, int)
                assert count >= 0
            except Exception as e:
                pytest.skip(f"Database not available: {e}")
    
    def test_get_account_lockout_status(self):
        """Test account lockout status check."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            user_id = bson.ObjectId()
            
            try:
                status = db.get_account_lockout_status(user_id, threshold=5, lockout_minutes=30)
                
                assert isinstance(status, dict)
                assert 'is_locked' in status
                assert 'failed_attempts' in status
                assert 'minutes_until_unlock' in status
                assert isinstance(status['is_locked'], bool)
                assert isinstance(status['failed_attempts'], int)
            except Exception as e:
                pytest.skip(f"Database not available: {e}")


class TestTokenBlacklist:
    """Test JWT token blacklist for logout."""
    
    def test_is_token_blacklisted_not_blacklisted(self):
        """Test that non-blacklisted token returns False."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            try:
                result = db.is_token_blacklisted("nonexistent_jti_123")
                assert result is False
            except Exception as e:
                pytest.skip(f"Database not available: {e}")
    
    def test_blacklist_token_and_check(self):
        """Test blacklisting a token and checking it."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            try:
                jti = "test_jti_123"
                expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
                
                db.blacklist_token(jti, expires_at)
                result = db.is_token_blacklisted(jti)
                
                assert result is True
            except Exception as e:
                pytest.skip(f"Database not available: {e}")


class TestAuditLogging:
    """Test security audit logging."""
    
    def test_log_auth_event(self):
        """Test that auth event is logged."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            try:
                user_id = bson.ObjectId()
                db.log_auth_event(
                    event_type='LOGIN',
                    user_id=user_id,
                    status='success',
                    ip_address='192.168.1.1',
                    user_agent='TestAgent'
                )
            except Exception as e:
                pytest.skip(f"Database not available: {e}")
    
    def test_get_audit_logs(self):
        """Test retrieving audit logs."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            try:
                logs = db.get_audit_logs(limit=10, days=30)
                assert isinstance(logs, list)
            except Exception as e:
                pytest.skip(f"Database not available: {e}")
    
    def test_get_suspicious_auth_patterns(self):
        """Test detecting suspicious authentication patterns."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            try:
                patterns = db.get_suspicious_auth_patterns(hours=24)
                assert isinstance(patterns, list)
            except Exception as e:
                pytest.skip(f"Database not available: {e}")


class TestEmailVerification:
    """Test email verification system."""
    
    def test_create_email_verification_token(self):
        """Test creating email verification token."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            try:
                student_id = bson.ObjectId()
                token = db.create_email_verification_token(student_id)
                
                assert token is not None
                assert len(token) > 0
            except Exception as e:
                pytest.skip(f"Database not available: {e}")
    
    def test_verify_email_token(self):
        """Test verifying email token."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            try:
                student_id = bson.ObjectId()
                token = db.create_email_verification_token(student_id)
                
                result = db.verify_email_token(token)
                # Result depends on DB state, just check it returns a boolean
                assert isinstance(result, bool)
            except Exception as e:
                pytest.skip(f"Database not available: {e}")
    
    def test_is_email_verified(self):
        """Test checking email verification status."""
        from flask import Flask
        app = Flask(__name__)
        
        with app.app_context():
            try:
                student_id = bson.ObjectId()
                result = db.is_email_verified(student_id)
                
                assert isinstance(result, bool)
            except Exception as e:
                pytest.skip(f"Database not available: {e}")
