// QR survey: visitor fills a short form, then gets a link to download the PDF.
// Submissions are appended to data/submissions.csv; the admin page downloads them.
const express = require('express');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const QRCode = require('qrcode');

const PORT = process.env.PORT || 3000;
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'change-me';
const PUBLIC_URL = process.env.PUBLIC_URL || `http://localhost:${PORT}`;
const PDF_FILE = process.env.PDF_FILE || path.join(__dirname, 'files', 'brochure.pdf');
const PDF_NAME = process.env.PDF_NAME || 'brochure.pdf';
const SECRET = process.env.TOKEN_SECRET || crypto.randomBytes(32).toString('hex');

const DATA_DIR = path.join(__dirname, 'data');
const CSV_FILE = path.join(DATA_DIR, 'submissions.csv');
const FIELDS = ['submitted_at', 'name', 'phone', 'company', 'email', 'pain_points'];

fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(CSV_FILE)) fs.writeFileSync(CSV_FILE, FIELDS.join(',') + '\n');

const csvCell = (v) => {
  let s = String(v ?? '').replace(/\r?\n/g, ' ').trim();
  if (/^[=+\-@]/.test(s)) s = "'" + s; // stop formula injection when opened in Excel
  return '"' + s.replace(/"/g, '""') + '"';
};
const sign = (id) => crypto.createHmac('sha256', SECRET).update(id).digest('hex').slice(0, 32);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

const app = express();
app.use(express.urlencoded({ extended: false, limit: '20kb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.post('/submit', (req, res) => {
  const b = req.body;
  const row = {
    submitted_at: new Date().toISOString(),
    name: (b.name || '').slice(0, 200),
    phone: (b.phone || '').slice(0, 50),
    company: (b.company || '').slice(0, 200),
    email: (b.email || '').slice(0, 200),
    pain_points: (b.pain_points || '').slice(0, 3000),
  };
  if (!row.name.trim() || !row.email.includes('@')) {
    return res.status(400).send(page('Missing details', '<p>Please go back and enter at least your name and a valid email.</p><a class="btn" href="/">Back</a>'));
  }
  fs.appendFileSync(CSV_FILE, FIELDS.map((f) => csvCell(row[f])).join(',') + '\n');
  const id = crypto.randomBytes(8).toString('hex');
  res.redirect(303, `/thanks?id=${id}&t=${sign(id)}`);
});

app.get('/thanks', (req, res) => {
  const { id = '', t = '' } = req.query;
  if (t !== sign(String(id))) return res.redirect('/');
  const link = `/download?id=${encodeURIComponent(id)}&t=${encodeURIComponent(t)}`;
  res.send(page('Thank you!', `
    <p>Thanks for sharing your details. Your document is ready.</p>
    <a class="btn" href="${esc(link)}">Download PDF</a>`));
});

app.get('/download', (req, res) => {
  const { id = '', t = '' } = req.query;
  if (t !== sign(String(id))) return res.redirect('/');
  if (!fs.existsSync(PDF_FILE)) return res.status(404).send('PDF not found on server.');
  res.download(PDF_FILE, PDF_NAME);
});

// Admin: QR code poster and CSV export, protected with HTTP basic auth.
const requireAdmin = (req, res, next) => {
  const [, encoded = ''] = (req.headers.authorization || '').split(' ');
  const [, pass = ''] = Buffer.from(encoded, 'base64').toString().split(':');
  if (pass === ADMIN_PASSWORD) return next();
  res.set('WWW-Authenticate', 'Basic realm="admin"').status(401).send('Login required');
};

app.get('/admin', requireAdmin, async (req, res) => {
  const count = fs.readFileSync(CSV_FILE, 'utf8').trim().split('\n').length - 1;
  const qr = await QRCode.toDataURL(PUBLIC_URL, { width: 600, margin: 2 });
  res.send(page('Admin', `
    <p><strong>${count}</strong> submission(s) so far.</p>
    <a class="btn" href="/admin/submissions.csv">Download submissions (CSV)</a>
    <h2>QR code</h2>
    <p>Points to <code>${esc(PUBLIC_URL)}</code></p>
    <img src="${qr}" alt="QR code" style="width:100%;max-width:320px">
    <p><a href="/admin/qr.png" download="survey-qr.png">Download QR as PNG</a></p>`));
});

app.get('/admin/submissions.csv', requireAdmin, (req, res) => res.download(CSV_FILE, 'submissions.csv'));

app.get('/admin/qr.png', requireAdmin, async (req, res) => {
  res.type('png').send(await QRCode.toBuffer(PUBLIC_URL, { width: 1200, margin: 2 }));
});

function page(title, body) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(title)}</title>
<link rel="stylesheet" href="/style.css"></head><body><main class="card"><h1>${esc(title)}</h1>${body}</main></body></html>`;
}

app.listen(PORT, () => console.log(`Survey running on port ${PORT} (public URL: ${PUBLIC_URL})`));
