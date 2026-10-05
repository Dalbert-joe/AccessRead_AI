export const API_URL=process.env.NEXT_PUBLIC_API_URL||'http://localhost:8000';
export async function processPdf(file:File){const fd=new FormData();fd.append('file',file);const r=await fetch(`${API_URL}/process/pdf`,{method:'POST',body:fd});if(!r.ok)throw new Error(await r.text());return r.json()}
export async function processUrl(url:string){const r=await fetch(`${API_URL}/process/url`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});if(!r.ok)throw new Error(await r.text());return r.json()}
