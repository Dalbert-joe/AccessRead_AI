from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, HttpUrl
from extraction.pdf import process_pdf
from extraction.web import process_url
from structure.semantic import build_document
from inference.runtime import get_inference_engine

router = APIRouter()

class TextRequest(BaseModel):
    text: str
    title: str | None = None

@router.post('/process/pdf')
async def pdf_endpoint(file: UploadFile = File(...)):
    if file.content_type not in {'application/pdf', 'application/octet-stream'} and not (file.filename or '').lower().endswith('.pdf'):
        raise HTTPException(400, 'Please upload a PDF file.')
    data = await file.read()
    if not data:
        raise HTTPException(400, 'The uploaded file is empty.')
    try:
        raw = process_pdf(data)
        return build_document(raw.text, raw.title, source='pdf', metadata={'ocr_pages': raw.ocr_pages, 'pages': raw.pages})
    except Exception as e:
        raise HTTPException(422, f'PDF processing failed: {e}')

@router.post('/process/url')
async def url_endpoint(payload: dict):
    url = payload.get('url', '')
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(400, 'URL must start with http:// or https://')
    try:
        raw = process_url(url)
        return build_document(raw.text, raw.title, source='web', metadata={'url': url})
    except Exception as e:
        raise HTTPException(422, f'Webpage processing failed: {e}')

@router.post('/process/text')
async def text_endpoint(payload: TextRequest):
    if not payload.text.strip():
        raise HTTPException(400, 'Text cannot be empty.')
    return build_document(payload.text, payload.title or 'Untitled document', source='text', metadata={})

@router.get('/inference/status')
def inference_status():
    engine = get_inference_engine()
    return engine.status()
