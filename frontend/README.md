# Deepfake Detection System — Frontend

## Overview

This folder contains the React single-page app for the Final Year Project *Detection of Deepfake Images, Videos, and Audio*. Users can upload media, run detection against the backend API, view results, and browse session history.

## Tech stack

- **React 18** with **Vite 5**
- **React Router** (app shell; main flow is on the home page)
- **Tailwind CSS** for styling
- **Axios** for HTTP, **react-dropzone** for uploads, **lucide-react** for icons

## Prerequisites

- **Node.js** 18+ (LTS recommended)
- A running **backend** that exposes the API expected by `src/services/api.js` (default base: `http://localhost:8000`, path prefix `/api/v1`)

## Setup & Installation

```bash
cd frontend
npm install
```

## Environment variables

| Variable        | Description                          | Default                 |
|----------------|---------------------------------------|-------------------------|
| `VITE_API_URL` | Backend origin (no trailing `/api`)   | `http://localhost:8000` |

Create a `.env` in this folder if the API is not on the default host/port:

```env
VITE_API_URL=http://localhost:8000
```

Vite only exposes variables prefixed with `VITE_` to the client.

## Scripts

| Command        | Description                    |
|----------------|--------------------------------|
| `npm run dev`  | Dev server (port **5173**, opens browser) |
| `npm run build`| Production build to `dist/`    |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | ESLint on `.js` / `.jsx`       |

## Development

1. Start the backend so `/api/v1` is reachable.
2. Run `npm run dev`.
3. Open the URL shown in the terminal (typically `http://localhost:5173`).

## Project layout (high level)

```
frontend/
├── src/
│   ├── App.jsx              # Layout, header, footer, router
│   ├── pages/HomePage.jsx   # Upload, analysis, results, history
│   ├── components/          # UI pieces (upload, dashboards, modals, etc.)
│   ├── services/api.js      # Axios client, detection & history endpoints
│   ├── hooks/useSession.js  # Session id for API calls
│   └── utils/formatters.js  # Display helpers
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```
