import { BlogPosts } from '@/app/components/posts'

export default function Page() {
  return (
    <section>
      <h1 className="mb-8 text-2xl font-semibold tracking-tighter">
        Niels
      </h1>
      <p className="mb-4">
        {`I'm just a guy. I like old-ass movies and older books. I play an unhealthy amount of soccer ever since my doctor forbade me from playing an unhealthy amount of hockey. I'm currently attending Fractal Bootcamp.`}
      </p>
      <div className="my-8">
        <BlogPosts />
      </div>
    </section>
  )
}
