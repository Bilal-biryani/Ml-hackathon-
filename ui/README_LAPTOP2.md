# PathSnap: Laptop 2 (evaluator UI) instructions

Team PT-040 · UE24CS352A ML Hackathon

This folder is the **complete Laptop 2 package**. Laptop 2 only displays results. All computation (model, HMM/Viterbi, scoring against the reference route) happens on **Laptop 1** inside the FastAPI server. Laptop 2 sends the selected trip id to Laptop 1 and draws what comes back.

> **For an AI assistant (e.g. Antigravity) helping on this laptop:** your job is only to **set up, run and test** this UI against the API on Laptop 1. Do **not** modify `index.html` or the scripts. Do **not** add hard-coded trip ids, results or routes. Do **not** download or add any model or data files. The competition rules forbid hard-coded outputs, and ground truth must never be on this laptop. If something fails, follow the troubleshooting table and report the exact error text.

---

## 0. What is in this folder

| File | Purpose |
|---|---|
| `index.html` | The evaluator UI: a single page using Leaflet and OpenStreetMap |
| `vendor/leaflet/` | Leaflet 1.9.4 bundled locally, so the page works without the Leaflet CDN |
| `start_ui.bat` | Starts a small local web server on port 5500 and opens the UI already connected to Laptop 1 |
| `check_api.bat` | Network test: ping Laptop 1, then call `/health` and `/trips` |
| `README_LAPTOP2.md` | This file |

The API on Laptop 1 provides:

| Method | URL | Purpose |
|---|---|---|
| GET | `http://<LAPTOP1_IP>:8000/health` | Model name and parameters (check the server is alive) |
| GET | `http://<LAPTOP1_IP>:8000/trips` | The 10 evaluation trips the server has |
| POST | `http://<LAPTOP1_IP>:8000/predict` with body `{"trip_id": "..."}` | Runs the full pipeline; returns GPS points, predicted route and metrics |

---

## 1. Requirements on Laptop 2

- Windows 10/11 with Chrome or Edge.
- **Optional:** Python 3 (any version) to run the local web server. Without Python you can open `index.html` directly (step 4b).
- `curl` and `ping` (built into Windows 10/11).
- Internet access is only needed for the street-map background tiles (OpenStreetMap). Points, routes and metrics work without it.

Check Python (optional):
```bash
python --version
```

---

## 2. Network: connect Laptop 1 and Laptop 2

Both laptops must be on the **same network**. Pick one option:

1. **Phone hotspot (recommended):** connect both laptops to the same mobile hotspot. Campus Wi-Fi often blocks laptop-to-laptop traffic.
2. **Same Wi-Fi:** works only if the network allows devices to see each other.
3. **Ethernet cable:** connect the laptops directly and set static IPs. On each laptop: Control Panel → Network and Sharing Center → Change adapter settings → Ethernet → Properties → IPv4 → "Use the following IP address":
   - Laptop 1: `192.168.50.1`, subnet `255.255.255.0`
   - Laptop 2: `192.168.50.2`, subnet `255.255.255.0`
   - Laptop 1's IP is then `192.168.50.1`.

**Get Laptop 1's IP:** on Laptop 1, run `ipconfig` and read the **IPv4 Address** of the Wi-Fi or Ethernet adapter that is connected (e.g. `192.168.43.25`). Below, this is called `LAPTOP1_IP`.

Laptop 1 must already be running the API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Its CMD window should show `Uvicorn running on http://0.0.0.0:8000`.

---

## 3. Test the connection (on Laptop 2)

Double-click **`check_api.bat`** and enter `LAPTOP1_IP`, or run it in CMD:
```bash
check_api.bat 192.168.43.25
```
Expected output:
1. `ping`: `Reply from 192.168.43.25 ...`
2. `/health`: `{"status":"ok","model":"LightGBM", ...}`
3. `/trips`: `{"count":10,"trips":[{"trip_id":"...","n_fixes":...}, ...]}`

Manual equivalent:
```bash
ping 192.168.43.25
curl http://192.168.43.25:8000/health
curl http://192.168.43.25:8000/trips
```
Optional test of one prediction from the command line (replace the trip id with one from `/trips`):
```bash
curl -X POST http://192.168.43.25:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\": \"00000000\"}"
```
It returns a long JSON containing `"metrics"` (Top-1 accuracy etc.).

**Do not continue until steps 2 and 3 return JSON.** If they don't, see section 7.

---

## 4. Start the UI

### 4a. With Python (recommended)
Double-click **`start_ui.bat`** and enter `LAPTOP1_IP`, or run:
```bash
start_ui.bat 192.168.43.25
```
This opens a window titled **"PathSnap UI server (keep open)"**. Leave it open. It also opens the browser at:
```
http://localhost:5500/?api=http://192.168.43.25:8000
```
The page connects automatically.

Manual equivalent:
```bash
cd /d <this folder>
python -m http.server 5500
```
Then open `http://localhost:5500` in the browser.

### 4b. Without Python
Double-click `index.html`. In the page, type `http://LAPTOP1_IP:8000` into **API address** and click **Connect and load trips**.

---

## 5. Use the UI (what the evaluator does)

1. **API address** must be `http://LAPTOP1_IP:8000`, including `http://` and `:8000`. It is remembered in the browser.
2. Click **Connect and load trips**. The status line shows `Connected · model LightGBM · 10 trips`, and the left panel lists the 10 trips.
3. Click a trip. The status shows `running pipeline for trip …`, and after about 1–5 seconds:
   - **Map:** blue dots are the GPS observations, the red line is the predicted route (HMM/Viterbi), green is start and black is end. The layer control (top right) toggles the layers, and the legend is bottom left.
   - **Cards:** Top-1 accuracy (HMM), route continuity, route F1, number of GPS fixes and server time. The small text under each card is the classifier-only value for comparison.
   - **Table:** classifier-only vs classifier + HMM for Top-1 accuracy, continuity, route precision, recall and F1, plus timings.
4. All metrics are computed **on Laptop 1** against its private reference route. The UI never receives the reference route.

Direct link that connects and runs one trip:
```
http://localhost:5500/?api=http://LAPTOP1_IP:8000&trip=<trip_id>
```

---

## 6. Full test checklist

- [ ] `check_api.bat` → ping OK, `/health` OK, `/trips` lists 10 trips
- [ ] UI connects: "Connected · model LightGBM · 10 trips"
- [ ] Trip 1 runs: map shows the route, cards show metrics
- [ ] Trip 2 runs
- [ ] Trip 3 runs
- [ ] Trip 4 runs
- [ ] Trip 5 runs
- [ ] Trip 6 runs
- [ ] Trip 7 runs
- [ ] Trip 8 runs
- [ ] Trip 9 runs
- [ ] Trip 10 runs. **All ten must work**, as the rules require.
- [ ] On **Laptop 1**, each click logs `PREDICT request trip=... from <Laptop-2-IP>` and `RESULT trip=... Top-1 ...`
- [ ] **Screenshot A** (Laptop 2): the UI with a trip loaded (map + metric cards visible)
- [ ] **Screenshot B** (Laptop 1): the CMD window showing the `PREDICT` / `RESULT` log lines
- [ ] Send both screenshots to the report owner, to paste into Section 7 of the report

Record results here while testing:

| Trip id | Top-1 (HMM) | Continuity | Route F1 | Server time (s) | OK? |
|---|---|---|---|---|---|
| | | | | | |

---

## 7. Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| `ping` → "Request timed out" or "Destination host unreachable" | Not on the same network, or the network blocks device-to-device traffic | Use a phone hotspot or an Ethernet cable (section 2). Re-check `LAPTOP1_IP` with `ipconfig` on Laptop 1. |
| ping OK, but `curl ...:8000/health` times out | Windows Firewall on Laptop 1 | On Laptop 1, CMD **as Administrator**: `netsh advfirewall firewall add rule name="PathSnap" dir=in action=allow protocol=TCP localport=8000` |
| `curl: (7) Failed to connect ... Connection refused` | API not running, or started without `--host 0.0.0.0` | Restart on Laptop 1: `uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| UI: "Cannot reach API: Failed to fetch" | Wrong address, or one of the causes above | Use `http://LAPTOP1_IP:8000`, not `localhost`; run `check_api.bat` |
| `check_api.bat` works, but the UI says "Failed to fetch" and Laptop 1's CMD shows **no** request | The browser blocked the request (Chrome/Edge local-network protection) | Open the UI from **`http://localhost:5500`** on Laptop 2 (what `start_ui.bat` does), not from an IP address. If the browser asks to "access devices on your local network", click **Allow**. |
| `/trips` → `"count":0` | Laptop 1 has no trips in `data\eval\` | On Laptop 1, copy the trip folders into `data\eval\` and restart the API |
| UI: `Error: HTTP 404` on a trip | Trip id not on the server | Click **Connect** again to reload the list |
| UI: `Error: HTTP 500` | Server error on Laptop 1 | Read the traceback in Laptop 1's CMD window and report it |
| Map background grey, but route visible | No internet on Laptop 2 (tiles come from OpenStreetMap) | Connect Laptop 2 to the internet; the results are still valid |
| `start_ui.bat`: "port 5500 already in use" | Old server still running | Close the old "PathSnap UI server" window, or run `python -m http.server 5501` and open `http://localhost:5501` |
| `python` not recognised | Python not installed or not on PATH | Use option 4b (open `index.html` directly) |
| IP changed (hotspot reconnected) | DHCP gave Laptop 1 a new IP | Run `ipconfig` on Laptop 1 again and re-enter the new address in the UI |

---

## 8. Evaluation-day procedure (short version)

1. Laptop 1: copy the evaluator's 10 trip folders into `data\eval\`, start the API, and keep its CMD window visible.
2. Laptop 2: run `check_api.bat`, then `start_ui.bat`.
3. Let the evaluator pick any trip in the UI.
4. Show the evaluator the API logs on Laptop 1 (the `PREDICT` / `RESULT` lines for their clicks).
