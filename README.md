# PBEL: ESP32-S3 Edge-AI Acoustic Event Classification and Smart Noise Monitoring

```
firmware/pbel_esp32s3/   main sketch (+ features.h, dashboard.h)  <- put YOUR model_data.h here
firmware/mic_test/       microphone-only test sketch
frontend/dashboard.html  standalone dashboard (double-click; Demo mode or live)
ml_training/             Colab notebook + the two training scripts
```

## Wiring (matches the sketch)
| INMP441 | ESP32-S3 | | OLED SSD1306 | ESP32-S3 |
|---|---|---|---|---|
| VDD | 3V3 | | VCC | 3V3 |
| GND | GND | | GND | GND |
| L/R | GND | | SDA | GPIO 17 |
| SCK | GPIO 4 | | SCL | GPIO 18 |
| WS  | GPIO 5 | | | |
| SD  | GPIO 6 | | | |

## 1. Arduino IDE setup (once)
- Boards Manager: **esp32 by Espressif 3.x**. Board **ESP32S3 Dev Module**, **USB CDC On Boot: Enabled**, PSRAM Disabled.
- Library Manager: **Chirale_TensorFlowLite**, **Adafruit SSD1306**, **Adafruit GFX**.
- Use the **native USB** port (it appears as its own COM port; select that one). Serial Monitor 115200.

## 2. Test the microphone first
Open `firmware/mic_test/mic_test.ino`, upload, clap near the mic. Expect `MIC WORKS on LEFT channel`.

## 3. Get your trained model
Run `ml_training/PBEL_Colab_Training.ipynb` in Google Colab (Runtime > Run all). Unzip the download, then:
- copy **model_data.h** into `firmware/pbel_esp32s3/`
- open **norm.json** and copy `mean` / `std` into `NORM_MEAN` / `NORM_STD` at the top of `pbel_esp32s3.ino`
- keep `CONF_THRESHOLD = 0.75f` (tune later)

## 4. Run the main sketch
Open `firmware/pbel_esp32s3/pbel_esp32s3.ino` (only ONE .ino file per folder), upload. Serial shows:
`[boot] ring + features ok` -> `Model ... arena used ...` -> `[boot] model ok` -> `[boot] I2S ok` -> `Dashboard: http://192.168.4.1` -> `[boot] ready`.
The last `[boot]` line printed tells you where a problem is.
Quiet room: note `idle rms=`; set `RMS_ON` about 3x that, `RMS_OFF` slightly lower; re-upload.

## 5. Dashboard
- With no router configured, the ESP32 creates a hotspot: join **PBEL-Monitor** (password **pbel1234**), open **http://192.168.4.1** (served by the board).
- Or open `frontend/dashboard.html` on the same device, enter `192.168.4.1`, press **Connect**. It starts in **Demo mode** with simulated data, so you can show the UI without hardware.
- To use your router/hotspot instead, set `WIFI_SSID` / `WIFI_PASS` (2.4 GHz); the OLED shows the IP.

## Numbers for the report (Serial Monitor)
Model size and `arena used` (startup), `STATS` line after each event (feature/inference time, min free heap), `EVENT CONFIRMED` lines (live trials, false-alarm run).

## Troubleshooting
- `model_data.h is missing` -> copy it into the sketch folder.
- `redefinition of ...` errors -> two .ino files in one folder.
- OLED shows `SET NORM_MEAN/STD` -> paste values from norm.json.
- Nothing in Serial Monitor -> select the native-USB COM port, CDC On Boot = Enabled, press RST.
- `raw24 min=0 max=0` -> mic wiring/solder (run mic_test).
- Limits: levels are relative RMS (not dB SPL); model trained on public ESC-50 data, so device-specific accuracy may differ.
