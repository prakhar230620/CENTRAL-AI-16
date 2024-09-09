# backend/models/database_models.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
from database.database import Base

class CommonFieldsMixin:
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.datetime.utcnow)

class User(Base, CommonFieldsMixin):
    __tablename__ = 'users'

    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)

    preferences = relationship('UserPreference', back_populates='user', uselist=False)
    api_keys = relationship('APIKey', back_populates='user')
    usage_stats = relationship('UsageStat', back_populates='user')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @validates('email')
    def validate_email(self, key, address):
        assert '@' in address, "Email address must contain '@'"
        return address
class UserPreference(Base, CommonFieldsMixin):
    __tablename__ = 'user_preferences'

    user_id = Column(Integer, ForeignKey('users.id'), unique=True, nullable=False)
    theme = Column(String(50), default='light')
    language = Column(String(50), default='en')

    user = relationship('User', back_populates='preferences')


class ServiceMetadata(Base, CommonFieldsMixin):
    __tablename__ = 'service_metadata'

    service_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    activation_status = Column(Boolean, default=False)

    credentials = relationship('ServiceCredential', back_populates='service')


class ServiceCredential(Base, CommonFieldsMixin):
    __tablename__ = 'service_credentials'

    service_id = Column(String(50), ForeignKey('service_metadata.service_id'), nullable=False)
    credential_data = Column(Text, nullable=False)

    service = relationship('ServiceMetadata', back_populates='credentials')


class APIKey(Base, CommonFieldsMixin):
    __tablename__ = 'api_keys'

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    key = Column(String(100), unique=True, nullable=False, index=True)
    encrypted_key = Column(String(256), nullable=False)

    user = relationship('User', back_populates='api_keys')


class UsageStat(Base, CommonFieldsMixin):
    __tablename__ = 'usage_stats'

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    service_id = Column(String(50), ForeignKey('service_metadata.service_id'), nullable=False)
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime, default=func.now())

    user = relationship('User', back_populates='usage_stats')
    service = relationship('ServiceMetadata')


class Log(Base, CommonFieldsMixin):
    __tablename__ = 'logs'

    log_id = Column(String(50), unique=True, nullable=False, index=True)
    log_data = Column(Text, nullable=False)
    log_level = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Example usage:
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# engine = create_engine('sqlite:///ai_system.db')
# Base.metadata.create_all(engine)
# Session = sessionmaker(bind=engine)
# session = Session()