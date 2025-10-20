from sqlmodel import Session, select
from app.db import engine
from app.models import Post
from app.db import init_db

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
    session.commit()
    print("Session committed")
    session.refresh(post)
    print("Post refreshed")
    print(f"Post created: {post.id}, {post.title}, {post.slug}, {post.content}")

    query = select(Post).where(Post.id == post.id)
    print("Query created")
    rows = session.exec(query).all()
    print("Rows fetched")
    print("Queried rows:", [(row.id, row.title) for row in rows])
    
if __name__ == "__main__":
  main()