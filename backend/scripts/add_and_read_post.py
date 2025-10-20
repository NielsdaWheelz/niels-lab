from sqlmodel import Session, select
from app.db import engine
from app.models import Post
from app.db import init_db
from sqlalchemy.exc import IntegrityError

def main():
  init_db()
  print("Database initialized")
  with Session(engine) as session:
    post = Post(
      title="Test Post",
      slug="test-post",
      content="This is a test post",
    )
    print("Post created")
    session.add(post)
    print("Post added to session")
    try:
      session.commit()
    except IntegrityError as err:
        session.rollback()
        if getattr(err.orig, "sqlstate", None) == "23505":  # unique violation
            print("Slug already exists")
        else:
            raise
    else:
        session.refresh(post)
        print(f"Post created: {post.id}, {post.title}, {post.slug}, {post.content}")



    query = select(Post)
    print("Query created")
    rows = session.exec(query).all()
    print("Rows fetched")
    print("Queried rows:", [(row.id, row.title) for row in rows])

  # with Session(engine) as session:
  #   post = session.exec(select(Post).where(Post.slug == "test-post")).first()
  #   if post:
  #       session.delete(post)
  #       session.commit()
  #       print("Post deleted")
  #   else:
  #       print("Post not found")
  #   query = select(Post)
  #   posts = session.exec(query).all()
  #   print("Posts fetched")
  #   print("Posts:", [(post.id, post.title) for post in posts])

if __name__ == "__main__":
  main()