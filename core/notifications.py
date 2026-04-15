"""Notification event helpers with optional SMTP email delivery."""

from __future__ import annotations

import smtplib
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import core.config as config
import core.database as database
from core.utils import setup_logging

logger = setup_logging()


def _should_send_notification(event_type: str, recipient: str) -> bool:
    """Check if notification should be sent based on rate limiting.
    
    Returns True if enough time has passed since last notification of this type
    to this recipient, False otherwise.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=config.NOTIFICATION_RATE_LIMIT_SECONDS)
    
    recent = database.get_db().notification_events.find_one(
        {
            "event_type": event_type,
            "recipient": recipient,
            "created_at": {"$gte": cutoff},
        },
        sort=[("created_at", -1)],
    )
    
    if recent:
        logger.debug(
            "Skipped notification (rate limited): %s to %s (last sent %s)",
            event_type, recipient, recent.get("created_at"),
        )
        return False
    
    return True


def _send_email(recipient: str, subject: str, body: str, html_body: str | None = None) -> bool:
    """Send email via SMTP. Returns True on success, False on failure."""
    if not config.NOTIFICATIONS_ENABLED:
        logger.debug("Notifications disabled; skipping email to %s", recipient)
        return False
    
    if not config.SMTP_SERVER or not config.SMTP_USERNAME:
        logger.warning("SMTP not configured; cannot send email to %s", recipient)
        return False
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{config.NOTIFICATION_FROM_NAME} <{config.NOTIFICATION_FROM_EMAIL}>"
        msg["To"] = recipient
        
        # Add plain text version (required)
        msg.attach(MIMEText(body, "plain"))
        
        # Add HTML version if provided (optional)
        if html_body:
            msg.attach(MIMEText(html_body, "html"))
        
        server = smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT)
        if config.SMTP_USE_TLS:
            server.starttls()
        
        if config.SMTP_USERNAME and config.SMTP_PASSWORD:
            server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
        
        server.sendmail(config.NOTIFICATION_FROM_EMAIL, [recipient], msg.as_string())
        server.quit()
        
        logger.info("Email sent successfully to %s: %s", recipient, subject)
        return True
    except smtplib.SMTPException as exc:
        logger.error("SMTP error sending to %s: %s", recipient, exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error sending email to %s: %s", recipient, exc)
        return False


def record_notification_event(event_type: str, recipient: str, subject: str, payload: dict) -> str:
    """Persist a notification event and optionally send via SMTP.
    
    Args:
        event_type: Type of notification (e.g., 'absence_alert', 'low_attendance_alert')
        recipient: Email address or identifier
        subject: Email subject line
        payload: Event data dictionary
    
    Returns:
        Notification event ID
    """
    mode = "production" if config.NOTIFICATIONS_ENABLED else "dry_run"
    
    doc = {
        "event_type": event_type,
        "recipient": recipient,
        "subject": subject,
        "payload": payload,
        "mode": mode,
        "sent": False,
        "created_at": datetime.now(timezone.utc),
    }
    
    # Check rate limiting to prevent email flooding
    if config.NOTIFICATIONS_ENABLED and not _should_send_notification(event_type, recipient):
        doc["mode"] = "production"
        doc["sent"] = False
        doc["suppressed"] = True
        logger.info("Notification suppressed by rate limit: %s to %s", event_type, recipient)
        result = database.get_db().notification_events.insert_one(doc)
        return str(result.inserted_id)
    
    # Attempt SMTP delivery if enabled and not rate limited
    if config.NOTIFICATIONS_ENABLED:
        # Generate plain text version
        body = f"Subject: {subject}\n\n"
        body += f"Event: {event_type}\n"
        body += f"Recipient: {recipient}\n"
        for key, value in payload.items():
            body += f"{key}: {value}\n"
        
        sent = _send_email(recipient, subject, body)
        doc["sent"] = sent
        if not sent:
            logger.warning("Failed to send email for %s to %s", event_type, recipient)
    
    result = database.get_db().notification_events.insert_one(doc)
    return str(result.inserted_id)


def get_notification_events(limit: int = 100) -> list[dict]:
    """Return recent notification events."""
    return list(
        database.get_db().notification_events.find({}).sort("created_at", -1).limit(max(1, limit))
    )
