# AccessRead AI

Real full-stack accessibility processing: PDF/webpage → extraction/OCR → semantic structure → reading-order reconstruction → accessible reader → simplification → browser Read Aloud.

## Architecture
- `frontend/`: Next.js + TypeScript + React + Tailwind, deployable to Vercel.
- `backend/`: FastAPI CPU inference server. PDFs and webpages are processed in memory and are not persisted.
- `backend/inference/runtime.py`: OpenVINO abstraction. When `models/semantic_classifier.xml` exists, it uses `ov.Core()`, `read_model()` and `compile_model(..., "CPU")` for real inference.

## Local setup
### Backend
Install Python 3.11+ and Tesseract OCR. On Windows, install Tesseract separately and ensure `tesseract.exe` is on PATH.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m inference.model_builder
uvicorn main:app --reload --port 8000
```

The model builder creates `backend/models/semantic_classifier.onnx` and OpenVINO IR files. The runtime loads the IR on CPU.

### Frontend
```powershell
cd frontend
npm install
npm run build
npm run dev
```

Open `http://localhost:3000`.

## API
- `GET /health`
- `POST /process/pdf` multipart field `file`
- `POST /process/url` JSON `{ "url": "https://..." }`
- `POST /process/text` JSON `{ "text": "...", "title": "..." }`

## Vercel
Set `NEXT_PUBLIC_API_URL` to the public HTTPS URL of the FastAPI service. Deploy only `frontend/` as the Vercel project root. Do not deploy OpenVINO to Vercel.

## Backend deployment
Use a Python-capable CPU host. Install the requirements, generate the model, then run:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Put HTTPS in front of the service and configure `CORS_ORIGINS` to the Vercel domain.

## Environment variables
- `NEXT_PUBLIC_API_URL`: public FastAPI HTTPS base URL.
- `CORS_ORIGINS`: comma-separated allowed frontend origins.

## Privacy
Uploaded PDFs are read into memory, processed, and discarded. No application database or permanent document storage is used.
