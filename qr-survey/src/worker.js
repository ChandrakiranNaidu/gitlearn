// QR survey on Cloudflare Workers: visitor fills a short form, then gets a link to download the PDF.
// Submissions are stored in a D1 database; /admin exports them as CSV and shows the QR code.
import qrcode from 'qrcode-generator';

const FIELDS = ['submitted_at', 'name', 'phone', 'company', 'email', 'pain_points'];
const LIMITS = { name: 200, phone: 50, company: 200, email: 200, pain_points: 3000 };

const SCHEMA = `CREATE TABLE IF NOT EXISTS submissions (
  id TEXT PRIMARY KEY, submitted_at TEXT NOT NULL,
  name TEXT, phone TEXT, company TEXT, email TEXT, pain_points TEXT)`;

let schemaReady = false;
async function db(env) {
  if (!schemaReady) {
    await env.DB.prepare(SCHEMA).run();
    schemaReady = true;
  }
  return env.DB;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (path === '/submit' && request.method === 'POST') return submit(request, env);
    if (path === '/thanks') return thanks(url, env);
    if (path === '/download') return download(url, env);
    if (path === '/admin' || path.startsWith('/admin/')) {
      const denied = checkAdmin(request, env);
      if (denied) return denied;
      if (path === '/admin/submissions.csv') return exportCsv(env);
      if (path === '/admin/qr.svg') return new Response(qrSvg(url.origin), { headers: { 'Content-Type': 'image/svg+xml' } });
      return admin(url, env);
    }
    // The PDF is only handed out through /download.
    if (path.startsWith('/files/')) return new Response('Not found', { status: 404 });
    return env.ASSETS.fetch(request);
  },
};

async function submit(request, env) {
  const form = await request.formData();
  const row = { submitted_at: new Date().toISOString() };
  for (const [field, max] of Object.entries(LIMITS)) row[field] = String(form.get(field) || '').trim().slice(0, max);

  if (!row.name || !row.email.includes('@')) {
    return html('Missing details', '<p>Please go back and enter at least your name and a valid email.</p><a class="btn" href="/">Back</a>', 400);
  }

  const id = crypto.randomUUID().replace(/-/g, '');
  await (await db(env))
    .prepare('INSERT INTO submissions (id, submitted_at, name, phone, company, email, pain_points) VALUES (?, ?, ?, ?, ?, ?, ?)')
    .bind(id, ...FIELDS.map((f) => row[f]))
    .run();
  return Response.redirect(new URL(`/thanks?id=${id}`, request.url), 303);
}

async function isValidId(env, id) {
  if (!/^[0-9a-f]{32}$/.test(id)) return false;
  return !!(await (await db(env)).prepare('SELECT 1 FROM submissions WHERE id = ?').bind(id).first());
}

async function thanks(url, env) {
  const id = url.searchParams.get('id') || '';
  if (!(await isValidId(env, id))) return Response.redirect(new URL('/', url), 302);
  return html('Thank you!', `
    <p>Thanks for sharing your details. Your document is ready.</p>
    <a class="btn" href="/download?id=${id}">Download PDF</a>`);
}

async function download(url, env) {
  const id = url.searchParams.get('id') || '';
  if (!(await isValidId(env, id))) return Response.redirect(new URL('/', url), 302);

  const file = env.PDF_FILE || 'brochure.pdf';
  const asset = await env.ASSETS.fetch(new URL(`/files/${file}`, url).toString());
  if (!asset.ok) return new Response('PDF not found. Put it in public/files/ and redeploy.', { status: 404 });

  const headers = new Headers(asset.headers);
  headers.set('Content-Type', 'application/pdf');
  headers.set('Content-Disposition', `attachment; filename="${file.replace(/"/g, '')}"`);
  return new Response(asset.body, { headers });
}

// Admin pages use HTTP basic auth: any username, password = ADMIN_PASSWORD secret.
function checkAdmin(request, env) {
  if (!env.ADMIN_PASSWORD) return new Response('Set the ADMIN_PASSWORD secret first (see README).', { status: 503 });
  const [scheme, encoded = ''] = (request.headers.get('Authorization') || '').split(' ');
  let password = '';
  try {
    password = atob(encoded).split(':').slice(1).join(':');
  } catch {}
  if (scheme === 'Basic' && password === env.ADMIN_PASSWORD) return null;
  return new Response('Login required', { status: 401, headers: { 'WWW-Authenticate': 'Basic realm="admin"' } });
}

async function admin(url, env) {
  const { count } = await (await db(env)).prepare('SELECT COUNT(*) AS count FROM submissions').first();
  return html('Admin', `
    <p><strong>${count}</strong> submission(s) so far.</p>
    <a class="btn" href="/admin/submissions.csv">Download submissions (CSV)</a>
    <h2>QR code</h2>
    <p>Points to <code>${esc(url.origin)}</code></p>
    <div class="qr">${qrSvg(url.origin)}</div>
    <p><a href="/admin/qr.svg" download="survey-qr.svg">Download QR code (SVG, prints sharp at any size)</a></p>`);
}

async function exportCsv(env) {
  const { results } = await (await db(env)).prepare('SELECT * FROM submissions ORDER BY submitted_at').all();
  const lines = [FIELDS.join(','), ...results.map((r) => FIELDS.map((f) => csvCell(r[f])).join(','))];
  // The BOM makes Excel read names with accents correctly.
  return new Response('﻿' + lines.join('\r\n') + '\r\n', {
    headers: {
      'Content-Type': 'text/csv; charset=utf-8',
      'Content-Disposition': 'attachment; filename="submissions.csv"',
    },
  });
}

function csvCell(value) {
  let s = String(value ?? '').replace(/\r?\n/g, ' ');
  if (/^[=+\-@]/.test(s)) s = "'" + s; // stop formula injection when opened in Excel
  return '"' + s.replace(/"/g, '""') + '"';
}

function qrSvg(text) {
  const qr = qrcode(0, 'M');
  qr.addData(text);
  qr.make();
  return qr.createSvgTag({ cellSize: 8, margin: 4, scalable: true });
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function html(title, body, status = 200) {
  return new Response(
    `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(title)}</title>
<link rel="stylesheet" href="/style.css"></head><body><main class="card"><h1>${esc(title)}</h1>${body}</main></body></html>`,
    { status, headers: { 'Content-Type': 'text/html; charset=utf-8' } },
  );
}
