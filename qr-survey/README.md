# QR Survey → PDF download (free, Cloudflare)

People scan a QR code, fill in a short form (Name, Phone, Airline/Company, Email, Pain points/interests),
and then download your PDF. Every submission is saved, and you can download them all as a spreadsheet (CSV).

| Page | What it does |
|------|--------------|
| `/` | The survey form (mobile friendly) |
| `/thanks` | Shown after submitting, with the **Download PDF** button |
| `/admin` | Password protected: submission count, CSV export, printable QR code |

## Why Cloudflare
- **Free:** the Workers free plan needs no credit card. 100,000 requests/day is far more than a few days of survey traffic.
- **Atlanta:** Cloudflare runs the site in the data center closest to each visitor. People scanning in Atlanta are served from Cloudflare's Atlanta data center.
  The small submissions database lives in one region; step 4 creates it in eastern North America (`enam`).

## Put it online (about 10 minutes, one time)

1. Install **Node.js LTS** (free): https://nodejs.org
2. Get this folder onto your computer (download the repo as ZIP from GitHub, or `git clone`), and open a terminal in the `qr-survey` folder.
3. Put **your PDF** at `public/files/brochure.pdf` (replace the placeholder).
   To use a different file name, change `PDF_FILE` in `wrangler.jsonc`.
4. Run these commands one by one:

   ```bash
   npm install
   npx wrangler login                                          # opens the browser: sign up / log in to Cloudflare (free, no card)
   npx wrangler d1 create qr-survey --location=enam --binding DB   # database in eastern North America, saved into wrangler.jsonc
   npx wrangler deploy                                         # publishes the site
   npx wrangler secret put ADMIN_PASSWORD                      # type the password you want for /admin
   ```

   - The first deploy may ask you to pick a `workers.dev` subdomain. Any name is fine.
   - At the end it prints your link, like `https://qr-survey.<your-subdomain>.workers.dev`.

5. Open `https://qr-survey.<your-subdomain>.workers.dev/admin` (any username, the password you just set).
   Click **Download QR code**, then print it or put it on your slides/banner.

## When the event is over
1. Go to `/admin` → **Download submissions (CSV)** and open it in Excel.
2. Take everything down:
   ```bash
   npx wrangler delete
   npx wrangler d1 delete qr-survey
   ```
   (Or leave it; the free plan does not expire or charge.)

## Changing things later
- Edit the form text in `public/index.html`, colors in `public/style.css`, or swap the PDF, then run `npx wrangler deploy` again.
- Test on your own computer first: create a file `.dev.vars` containing `ADMIN_PASSWORD=test`, then run `npm run dev` and open http://localhost:8787

## Troubleshooting
- **Check the database was linked:** after the `d1 create` command, `wrangler.jsonc` should contain a
  `"database_id"` line. If not, copy the `database_id` the command printed into the `d1_databases` entry
  (`"database_id": "..."`) and deploy again. (If you skip `d1 create`, `deploy` still creates a database,
  just in the region nearest to you instead of eastern North America.)
- **/admin says "Set the ADMIN_PASSWORD secret first"**: run step 4's last command again.
