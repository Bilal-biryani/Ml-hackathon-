@echo off
REM Optional (Laptop 1): allow incoming connections on port 8000 for the PathSnap API.
REM Right-click this file and choose "Run as administrator". Remove the rule after the demo with:
REM   netsh advfirewall firewall delete rule name="PathSnap API 8000"
netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000 profile=any
pause
