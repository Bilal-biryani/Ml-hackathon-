"""Command-line client for the PathSnap API (handy for testing from any laptop).

    python api/client.py --api http://192.168.1.10:8000            # list trips
    python api/client.py --api http://192.168.1.10:8000 --trip 00000000
"""
import argparse
import json
import urllib.request


def call(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--trip")
    a = ap.parse_args()
    if not a.trip:
        print(json.dumps(call(f"{a.api}/trips"), indent=2))
        return
    r = call(f"{a.api}/predict", {"trip_id": a.trip})
    short = {k: v for k, v in r.items() if k not in ("gps", "predicted_route", "predicted_arc_per_fix", "predicted_arc_sequence")}
    short["predicted_route_segments"] = len(r["predicted_route"])
    short["predicted_arcs"] = len(r["predicted_arc_sequence"])
    print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
