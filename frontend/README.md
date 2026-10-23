This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app), serving as the primary frontend for the F1 Race Winner Prediction project. It reads all data from the FastAPI backend in the repo root (`app/api.py`) — it has no database or data files of its own.

## Getting Started

**1. Start the backend with CORS enabled for this frontend's origin.** The API defaults to *denying all* cross-origin browser requests (a deliberate security default, Decision 044) — without this, every client-side fetch on this frontend fails with a generic `Failed to fetch` error, even though the API itself is running fine (confirm with `curl`, which isn't subject to CORS, to tell the two apart). From the repo root:

```bash
F1_CORS_ALLOW_ORIGINS=http://localhost:3000 python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

`scripts/dev.py` / `make dev` do **not** set this — that launcher starts the separate Streamlit dashboard (`app/dashboard.py`), not this Next.js app, and Streamlit's server-side HTTP client is never subject to browser CORS in the first place. Starting the backend via `make dev` first and this frontend second will reproduce the `Failed to fetch` failure.

**2. Confirm `frontend/.env.local` exists** (copy it if not) with `NEXT_PUBLIC_API_URL=http://localhost:8000` — see that file's own comment for the matching origin the backend needs to allow.

**3. Run the frontend's development server:**

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
