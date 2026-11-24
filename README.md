# ApisSonic — Acoustic Beehive Diagnostics

Rebuilt from the "Complete Architecture & Rebuild Blueprint" (Phases 1–5). Zero-build ES modules:
no bundler, no dependencies. Serve the folder over HTTPS and open it on a phone.

> Screening aid only. The bundled classifier weights were trained on **synthesized** colony
> sounds, not real hives. Confirm any reading with a physical inspection, and retrain on your
> own recordings (see "Retraining") before relying on it.

## Put it on an iPhone (free, via GitHub Pages)

1. Create a new **public** repository on github.com (e.g. `apissonic`).
2. Upload everything in this folder (drag the files into "Add file → Upload files"), or:
   ```bash
   git init && git add . && git commit -m "ApisSonic"
   git branch -M main
   git remote add origin https://github.com/<you>/apissonic.git
   git push -u origin main
   ```
3. In the repo: **Settings → Pages → Source: Deploy from a branch → main / (root) → Save**.
4. After a minute, open `https://<you>.github.io/apissonic/` in **Safari** on the iPhone.
5. Tap **Start listening**, allow the microphone, then **Share → Add to Home Screen**.

Any static HTTPS host works the same way (Cloudflare Pages, Netlify — `netlify.toml` is included).

## Run locally

```bash
python3 -m http.server 8080        # or: npx http-server . -p 8080
# http://localhost:8080/            the app (mic works on localhost)
# http://localhost:8080/test-suite.html   17 self-tests
node tests/dsp-ml-tests.js         # same 17 tests in Node
```

Phone on the same Wi-Fi without hosting: `/usr/bin/ruby ./start-phone-server.rb 8443`
(self-signed certificate; accept the browser warning).

## Colonies and saved scan images

- **Hive ID** is a dropdown of your colonies. Pick the colony you are listening to before the
  15 s scan; the scan is saved under that colony. Choose **＋ Add new colony…** to register a
  new one, and **✕** to remove a colony that has no saved scans. Colonies with saved scans
  and colonies opened from a QR tag are listed automatically.
- Every saved 15 s scan also stores three PNG images on the device (IndexedDB `scan_images`):
  the mean **spectrum** of the clean frames, the **waterfall** of the whole scan, and the
  **MFCC** cepstrogram. Rejected frames are marked red in the waterfall and MFCC images.
- **Yard triage** shows thumbnails of each hive's latest images; tap one to view it full size
  and save it (or all three). Older scans' images open from the **🖼** button in the log.

## First check on a real phone

Open **Calibration & model** while listening with the microphone:

- **Sample rate** — iPhones usually report 48000 Hz. The app adapts automatically.
- **Voice processing** — if it says "still ON", iOS ignored the request to disable echo
  cancellation / noise suppression / auto gain, and steady hums will be attenuated.
- Play a steady 250 Hz tone from another device: the peak should read ~250 Hz and hold.

## Files

| File | Purpose |
|---|---|
| `dsp-engine.js` | Web Audio graph, 4096-pt FFT, band power, transient rejection, 15 s scan, synthesizer |
| `ml-classifier.js` | 24 mel bands, 13 MFCCs, roll-off / flux / ZCR, 18→8→5 MLP, Bayesian band-ratio, 60/40 fusion |
| `calibration.js` | Wall-material gains, temperature / night priors, mic noise floor |
| `audio-store.js` | 16-bit mono WAV encoder, IndexedDB (`inspections`, `wav_clips`, `scan_images`) |
| `scan-images.js` | Records each 15 s scan and renders the spectrum, waterfall and MFCC PNGs saved for Yard triage |
| `qr-report.js` | Deep links, QR generator, yard triage ranking, printable report |
| `app.js`, `index.html`, `styles.css` | Mobile UI |
| `ml/train_hive_classifier.py` | Trainer (numpy) → `ml/custom_mlp_weights.json` |
| `sw.js`, `manifest.json`, `icons/` | Offline PWA |

## Retraining

1. Save scans, then download clips with **⬇ .wav** in the Inspection Log.
2. Sort them into `recordings/<label>/*.wav` with labels
   `healthy`, `preswarm`, `queenless`, `piping`, `fanning`.
3. `python3 ml/train_hive_classifier.py --data-dir recordings --out ml/custom_mlp_weights.json`
   (or `--synthetic-demo` to regenerate the bundled weights).
4. In the app: **📂 Load Weights JSON**.

## Native build (optional, untested here)

```bash
npm install
npm run build:www          # copies the web files into www/
npx cap add ios && npx cap sync && npx cap open ios
```
Add `NSMicrophoneUsageDescription` to `ios/App/App/Info.plist` (and `RECORD_AUDIO` to the
Android manifest) after the platform folders are generated. Check the Capacitor version in
`package.json` against the current release before installing.

## Where this build deliberately differs from the blueprint

- **Sample rate is read from the device**, not fixed at 44.1 kHz. WAV clips are saved at half
  the device rate (22.05 kHz, or 24 kHz on a 48 kHz iPhone) with a matching header.
- **QR tags are not always 21×21.** A 21×21 code holds 17 bytes, too few for an https link, so
  the generator uses the smallest size that fits (up to 41×41).
- **Roll-off, centroid and flux are computed over 60–2000 Hz** so high-frequency mic hiss
  does not skew them.
- **Input normalisation is folded into W1 / b1**, so the weights file is still just W1, b1, W2, b2.
- Capacitor's `webDir` is `www/` (built by `npm run build:www`) rather than the project root.
