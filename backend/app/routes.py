from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from .db import get_session
from .models import Post

