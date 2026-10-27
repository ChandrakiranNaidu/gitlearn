# QR Survey → PDF download

People scan a QR code, fill in a short form (Name, Phone, Airline/Company, Email, Pain points/interests),
and then download your PDF. Every submission is saved to a CSV you can download from the admin page.

| Page | What it does |
|------|--------------|
| `/` | The survey form (mobile friendly) |
| `/thanks` | Shown after submitting, with the **Download PDF** button |
| `/admin` | Password protected: submission count, CSV export, printable QR code |

## 1. Add your PDF
Replace `files/brochure.pdf` with your document (keep the name, or set `PDF_FILE` / `PDF_NAME`). Commit and push.

## 2. Host it in Atlanta, GA (2–3 days)
Recommended: **Vultr – Atlanta** (hourly billing; the $5–6/month plan costs well under $1 for 3 days).
Linode/Akamai **Atlanta, GA** works the same way.

1. Create an account at vultr.com → **Deploy New Server**.
2. Type: *Cloud Compute – Shared CPU*; Location: **Atlanta**; Image: **Ubuntu 24.04**; smallest plan (1 GB RAM).
3. Under **Startup Script**, paste `deploy/startup-script.sh` after editing the values at the top
   (`GITHUB_TOKEN` is needed only while the repo is private; `ADMIN_PASSWORD` is your admin password).
4. Deploy. After ~3–5 minutes the site is at `https://<server-ip-with-dashes>.sslip.io`
   (e.g. IP `45.32.10.5` → `https://45-32-10-5.sslip.io`). SSH in and `cat /root/SURVEY_URL.txt` to see it.
5. Open `/admin` (any username + your password) → **Download QR as PNG** and print it.

**When you're done:** download the CSV from `/admin`, then **Destroy** the server in Vultr
(stopping it still bills; only destroying ends charges).

## Run locally
```bash
npm install
ADMIN_PASSWORD=secret npm start   # http://localhost:3000
```

## Settings (environment variables)
| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `3000` | Port to listen on |
| `PUBLIC_URL` | `http://localhost:PORT` | URL encoded in the QR code |
| `ADMIN_PASSWORD` | `change-me` | Password for `/admin` |
| `PDF_FILE` / `PDF_NAME` | `files/brochure.pdf` | PDF served and its download filename |
| `TOKEN_SECRET` | random per start | Signs download links |

Submissions are stored in `data/submissions.csv` on the server (not in git).
