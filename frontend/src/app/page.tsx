import { api } from "@/lib/api"
import type { Post } from "@/types/types"

const HomePage = async () => {
  const posts = await api<Post[]>("/api/posts");

  return (
    <main className="mx-auto max-w-2xl p-8 space-y-6">
    <h1 className="text-3xl font-bold">Fractal Lab</h1>
    <ul className="space-y-4">
      {posts.map((p) => (
        <li key={p.id} className="border p-4 rounded-lg">
          <a
            href={`/posts/${p.slug}`}
            className="text-lg font-medium underline"
          >
            {p.title}
          </a>
        </li>
      ))}
    </ul>
  </main>
  )
}
export default HomePage