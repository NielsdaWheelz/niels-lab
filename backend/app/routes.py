from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from .db import get_session
from .models import Post, PostBase

router = APIRouter()


@router.get("/posts")
def list_posts(session: Session = Depends(get_session)):
  statement = select(Post).order_by(Post.created_at.desc())
  return session.exec(statement).all()

@router.get("/posts/{slug}")
def get_post(slug: str, session: Session = Depends(get_session)):
  statement = select(Post).where(Post.slug == slug)

  post = session.exec(statement).first()

  if not post:
    raise HTTPException(status_code=404, detail="Post not found")
  
  return post

@router.post("/posts", status_code=201)
def create_post(data: PostBase, session: Session = Depends(get_session)):
  post = Post(**data.model_dump())
  session.add(post)

  try:
    session.commit()
  except IntegrityError as err:
    session.rollback()
    raise HTTPException(status_code=400, detail=str(err.orig)) from err

  session.refresh(post)
  return post
