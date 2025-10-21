export const api = async <T>(path: string, init?: RequestInit): Promise<T> => {
  const base = process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "");
  if (!base) {
    console.error("NEXT_PUBLIC_API_URL is not defined in the runtime environment");
    throw new Error("NEXT_PUBLIC_API_URL missing");
  }

  const url = `${base}${path.startsWith("/") ? path : `/${path}`}`;

  try {
    const res = await fetch(url, {
      cache: "no-store",
      ...init,
    });

    if (!res.ok) {
      console.error("API request failed", res.status, res.statusText, url);
      throw new Error(`API ${res.status}`);
    }

    return res.json();
  } catch (error) {
    console.error("API request threw before response", url, error);
    throw error;
  }
}
