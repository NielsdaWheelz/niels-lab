const BASE = process.env.NEXT_PUBLIC_API_URL

export const api = async <T>(path: string, init?: RequestInit): Promise<T> => {
  const res = await fetch(`${BASE}${path}`, {
    cache: "no-store",
    ...init,
  });
  if (!res.ok) throw new Error(`API ${res.status}`)
    return res.json()
}