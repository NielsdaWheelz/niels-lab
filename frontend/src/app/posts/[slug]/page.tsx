import { api } from "@/lib/api";
import type { Post } from "@/types/types";

const PostPage = async ({ params }: { params: Promise<{ slug: string }> }) => {
  const { slug } = await params
  try {
    const post = await api<Post>(`/api/posts/${slug}`)
    return (
      <main className="mx-auto max-w-2xl p-8 space-y-6">
        <h1 className="text-3xl font-bold">{post.title}</h1>
        <p className="text-gray-400 text-sm">
          {new Date(post.created_at).toLocaleString()}
        </p>
        <article className="prose prose-invert">
          {post.content}
        </article>
      </main>
    )
  }
  catch (error) {
    return (
      <main className="mx-auto max-w-2xl p-8 space-y-6">
        <h1 className="text-3xl font-bold">Post not found</h1>
      </main>
    )
  }


}
export default PostPage
