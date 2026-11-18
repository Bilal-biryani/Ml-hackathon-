# PathSnap: two-laptop setup and test guide

| Laptop | Role | Needs |
|---|---|---|
| **Laptop 1** | Model + FastAPI backend. It holds the trips and reference routes. | Python 3.11, the full `pathsnap` folder |
| **Laptop 2** | Evaluator UI | A web browser, the `ui` folder, and Python only to serve the page |

Both laptops must be on the **same network**: a phone hotspot (recommended), the same Wi-Fi, or an Ethernet cable.

---

## Laptop 1: model + API

### 1. Copy the project
Copy the whole `pathsnap` folder (USB stick or zip) to, for example, `D:\pathsnap`. It must contain:
```
models\final_model.joblib, models\final_config.json, models\hmm_config.json
src\, api\, data\raw\student_test\ (for rehearsal), data\eval\ (evaluator trips go here)
```

### 2. Install Python and the dependencies (one time)
Install Python 3.11 from python.org and tick **"Add python.exe to PATH"**. Then open CMD:
```bash
cd /d D:\pathsnap
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Put the evaluator trips in place
Copy the evaluator's 10 trip folders into `data\eval\`, so you get `data\eval\00000015\00000015.track` and so on. If they arrive inside an extra folder (for example `data\eval\evaluator_test\00000015\...`), that also works: the API looks up to two folders down.

### 4. Start the API (one click)
Double-click **`start_api.bat`** in the project folder, or run it from CMD:
| Command | Serves |
|---|---|
| `start_api.bat` | `data\eval\` (the evaluation) |
| `start_api.bat rehearsal` | the 10 student-test trips (practice) |
| `start_api.bat "E:\some\folder"` | trip folders from another location |
It activates `.venv` if there is one, checks the packages and the model, prints this laptop's IP addresses, and starts `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Wait for these lines:
```
Loaded model LightGBM with 20 features and HMM {...}
Serving trips from ...\data\eval (10 trips)
UI on Laptop 2 should use:  http://<IP>:8000
Uvicorn running on http://0.0.0.0:8000
```
**Keep this window open and visible.** These are the API logs the evaluator must see. To stop it, press Ctrl+C, then `Y`.

### 5. Firewall
If Windows shows a popup for **Python**, click **Allow access** and tick **Private and Public**. If there was no popup and Laptop 2 can't connect, right-click **`open_firewall.bat`** and choose **Run as administrator** (it adds one inbound rule for port 8000).

### 6. Find Laptop 1's IP
`start_api.bat` prints it. Otherwise run `ipconfig` and note the **IPv4 Address** of the Wi-Fi or Ethernet adapter, e.g. `192.168.43.25`. Ignore `169.254.x.x` addresses; those belong to unconnected adapters.

### 7. Self-test on Laptop 1 (in a second CMD window)
```bash
curl http://localhost:8000/health
curl http://localhost:8000/trips
python api\client.py --api http://localhost:8000 --trip 00000000
```
The last command prints the metrics JSON (Top-1 accuracy etc.). Use any trip id that `/trips` lists.

---

## Laptop 2: UI

### 1. Copy only the `ui` folder
Copy it to, for example, `D:\pathsnap_ui`. It must contain `index.html` and `vendor\leaflet\...`.

### 2. Serve the page
Double-click **`start_ui.bat`** and type Laptop 1's IP when asked (or run `start_ui.bat 192.168.43.25`). It serves the page on `http://localhost:5500` and opens the browser already pointed at the API. To do it by hand:
```bash
cd /d D:\pathsnap_ui
python -m http.server 5500
```
then open **http://localhost:5500** in Chrome or Edge. Always open the page from `localhost`: Chrome and Edge block a page loaded from an IP address from calling another IP address.

### 3. Connect
1. In **API address**, type `http://<Laptop-1-IP>:8000`, e.g. `http://192.168.43.25:8000`.
2. Click **Connect and load trips**. The status should say `Connected · model LightGBM · 10 trips`.
3. Click any trip. The map shows the road network (grey), GPS points (blue) and predicted route (red), and the cards show Top-1 accuracy, continuity and route F1 (HMM vs classifier only).

Shortcut link that connects and runs a trip directly:
```
http://localhost:5500/?api=http://<Laptop-1-IP>:8000&trip=00000000
```

Map tiles (the street background) load from OpenStreetMap, so Laptop 2 needs internet for the background. Without internet, the road network, GPS points and route are still drawn, because they come from the API.

---

## Test checklist

- [ ] Both laptops are on the same hotspot or Wi-Fi.
- [ ] Laptop 2 CMD: `ping <Laptop-1-IP>` gets replies. If not, see troubleshooting.
- [ ] Laptop 2 CMD: `curl http://<Laptop-1-IP>:8000/health` returns `{"status":"ok",...}`.
- [ ] The UI connects and lists all 10 trips.
- [ ] **Every one** of the 10 trips runs and shows a route and metrics. The rules require all ten to work.
- [ ] For each click, Laptop 1's CMD shows `PREDICT request trip=... from <Laptop-2-IP>` and `RESULT trip=... Top-1 ...`.
- [ ] Screenshot 1: the UI on Laptop 2 with a trip loaded.
- [ ] Screenshot 2: Laptop 1's CMD with the request and result log lines.
- [ ] Both screenshots are pasted into the report (the orange boxes in section 7).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ping` fails or times out | The network blocks device-to-device traffic (common on campus Wi-Fi). Use a phone hotspot or an Ethernet cable. |
| ping works but `curl ...:8000` times out | Firewall on Laptop 1: run the `netsh` command in step 5 as Administrator. |
| `Connection refused` | The API isn't running, or was started without `--host 0.0.0.0` (`start_api.bat` sets it). |
| UI says "Cannot reach API" | Check the address: it must include `http://` and `:8000`, and use Laptop 1's IP, not `localhost`. |
| `/trips` returns `count: 0` | The trip folders aren't in `data\eval\` (or in `PATHSNAP_TRIPS_DIR`). Each trip needs its own folder named after the trip id. |
| `FileNotFoundError: final_model.joblib` | The `models\` folder wasn't copied to Laptop 1. |
| `uvicorn` not recognised | Activate the venv first (`.venv\Scripts\activate`), or use `python -m uvicorn ...`. |
| IP changed after reconnecting | Run `ipconfig` again on Laptop 1 and re-enter the address in the UI. |

### Laptop 1 as the hotspot (no phone, no usable Wi-Fi)
On Laptop 1: Settings → Network & Internet → **Mobile hotspot** → turn it on (share over Wi-Fi). Connect Laptop 2 to that network. Laptop 1's address is then usually `192.168.137.1` (check with `ipconfig`), so use `http://192.168.137.1:8000`.

### Wired option (no Wi-Fi)
Connect the laptops with an Ethernet cable. On each laptop, open Control Panel → Network and Sharing Center → Change adapter settings → Ethernet → Properties → IPv4 → "Use the following IP address":
- Laptop 1: `192.168.50.1`, mask `255.255.255.0`
- Laptop 2: `192.168.50.2`, mask `255.255.255.0`

Then use `http://192.168.50.1:8000` as the API address.
