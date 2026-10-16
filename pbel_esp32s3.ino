/*
  PBEL - ESP32-S3 Edge-AI Acoustic Event Classification and Smart Noise Monitoring
  INMP441 (I2S, 16 kHz) -> RMS trigger -> log-Mel -> Tiny CNN (TFLite Micro, int8)
  -> confidence + 3-of-5 confirmation -> OLED + Wi-Fi event list.   Core inference needs no internet.

  Needs in this folder:  features.h  and  model_data.h (from training; defines g_model, g_model_len)
  Libraries: Chirale_TensorFlowLite, Adafruit SSD1306, Adafruit GFX   (Board: ESP32 core 3.x, "ESP32S3 Dev Module")
*/
#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <time.h>
#include "driver/i2s_std.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "features.h"
#if !__has_include("model_data.h")
#error "model_data.h is missing: copy it from the Colab output (pbel_outputs.zip) into this sketch folder."
#endif
#include "model_data.h"
#include "dashboard.h"
#include "esp_heap_caps.h"

// ======================= USER SETTINGS =======================
// Paste from norm.json (training output):
const float NORM_MEAN = 0.0f;          // <-- MUST CHANGE: "mean" from norm.json
const float NORM_STD  = 1.0f;          // <-- MUST CHANGE: "std" from norm.json
const float CONF_THRESHOLD = 0.75f;    // start here; raise if false alarms, lower if events are missed
// Event trigger (relative RMS, full scale = 1.0). Watch the serial RMS in a quiet room and tune.
const float RMS_ON  = 0.020f;          // start event above this
const float RMS_OFF = 0.012f;          // end event below this for HANG_BLOCKS blocks
const int   HANG_BLOCKS = 3;           // 3 x 0.25 s
// Wi-Fi (optional; leave SSID empty to run fully offline)
const char* WIFI_SSID = "";
const char* WIFI_PASS = "";
// If no router is set / reachable, the ESP32 makes its own hotspot: join it and open http://192.168.4.1
const bool AP_FALLBACK = true;
const char* AP_SSID = "PBEL-Monitor";
const char* AP_PASS = "pbel1234";          // at least 8 characters
// Pins
const int PIN_BCLK = 4, PIN_WS = 5, PIN_SD = 6;      // INMP441 SCK, WS, SD  (L/R -> GND)
const int PIN_SDA = 17, PIN_SCL = 18;               // SSD1306 128x64 I2C
const uint8_t OLED_ADDR = 0x3C;
// Microphone channel: INMP441 L/R pin to GND -> left slot (false). L/R pin to 3V3 -> right slot (true).
// If you only ever read zeros, flip this to true and re-upload.
const bool MIC_RIGHT_SLOT = false;     // mic test confirmed: LEFT channel (L/R pin on GND)
const bool DEBUG_RAW = true;           // prints raw min/max sample values once per second
// =============================================================

const char* CLASS_NAME[4] = {"Normal Ambient", "Vehicle Horn", "Construction/Drill", "Emergency Siren"};
const char* CLASS_PRIO[4] = {"-", "Disturbance", "Disturbance", "Alert"};
constexpr int HOP = 4000;                 // 0.25 s between classifications
constexpr int RING = 24576;               // 1.5 s of audio (> 1 s window + processing slack)
constexpr int VOTES = 5, NEED = 3;        // 3-of-5 confirmation
constexpr int ARENA = 56 * 1024;           // tensor arena (checked at startup; raise if AllocateTensors fails)

static int16_t* ring;                     // allocated at runtime (keeps static RAM small)
static volatile uint32_t totalSamples = 0;
static volatile int32_t rawMin = 0, rawMax = 0;
static int16_t win[feat::WIN];
static float fvec[feat::FRAMES * feat::NMELS];
static uint8_t* arena;

static i2s_chan_handle_t rxh;
static Adafruit_SSD1306 oled(128, 64, &Wire, -1);
static WebServer server(80);
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter* interp;
static TfLiteTensor *tin, *tout;
static bool wifiOn = false;

struct Evt { uint32_t startMs; time_t epoch; float dur, conf; uint8_t cls; uint16_t id; };
static Evt hist[20]; static int histN = 0;
static uint16_t evSeq = 0; static uint16_t evCount[4] = {0, 0, 0, 0};
static float lastProbs[4] = {0, 0, 0, 0}, liveRms = 0; static uint32_t lastTf = 0, lastTi = 0;
static String ipStr;

// ---------------- audio capture task (core 0) ----------------
void audioTask(void*) {
  static int32_t buf[512];                                      // 256 stereo frames (L,R,L,R...)
  const int ch = MIC_RIGHT_SLOT ? 1 : 0;
  float xPrev = 0, yPrev = 0;                                   // DC blocker state (removes the mic's constant offset)
  for (;;) {
    size_t got = 0;
    if (i2s_channel_read(rxh, buf, sizeof(buf), &got, portMAX_DELAY) != ESP_OK) continue;
    int n = got / 8;
    uint32_t w = totalSamples;
    int32_t mn = 0, mx = 0;
    for (int i = 0; i < n; i++) {
      int32_t v = buf[2 * i + ch] >> 8;                          // 24-bit sample
      if (v < mn) mn = v; if (v > mx) mx = v;
      float x = (float)v, y = x - xPrev + 0.995f * yPrev;       // high-pass: y[n] = x[n] - x[n-1] + 0.995*y[n-1]
      xPrev = x; yPrev = y;
      float o = y / 256.0f;                                     // 24-bit -> 16-bit scale
      ring[(w + i) % RING] = (int16_t)(o > 32767.0f ? 32767 : (o < -32768.0f ? -32768 : o));
    }
    rawMin = mn; rawMax = mx;
    totalSamples = w + n;
  }
}

void i2sBegin() {
  i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  esp_err_t e1 = i2s_new_channel(&cc, NULL, &rxh);
  i2s_std_config_t sc = {
    .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = { .mclk = I2S_GPIO_UNUSED, .bclk = (gpio_num_t)PIN_BCLK, .ws = (gpio_num_t)PIN_WS,
                  .dout = I2S_GPIO_UNUSED, .din = (gpio_num_t)PIN_SD, .invert_flags = {} },
  };
  esp_err_t e2 = i2s_channel_init_std_mode(rxh, &sc);
  esp_err_t e3 = i2s_channel_enable(rxh);
  Serial.printf("I2S pins SCK=%d WS=%d SD=%d | setup codes (must be 0): %d %d %d\n", PIN_BCLK, PIN_WS, PIN_SD, (int)e1, (int)e2, (int)e3);
}

// ---------------- model ----------------
void modelBegin() {
  arena = (uint8_t*)heap_caps_aligned_alloc(16, ARENA, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  if (!arena) { Serial.println("Arena allocation FAILED (lower ARENA)"); for (;;) delay(1000); }
  const tflite::Model* m = tflite::GetModel(g_model);
  static tflite::MicroInterpreter si(m, resolver, arena, ARENA);
  interp = &si;
  if (interp->AllocateTensors() != kTfLiteOk) { Serial.println("AllocateTensors FAILED (increase ARENA?)"); for (;;) delay(1000); }
  tin = interp->input(0); tout = interp->output(0);
  Serial.printf("Model %u bytes, arena used %u of %u bytes (peak tensor RAM), free heap %u\n", g_model_len, (unsigned)interp->arena_used_bytes(), ARENA, (unsigned)ESP.getFreeHeap());
}

// returns class, sets conf
int classify(float* conf, uint32_t* tFeat, uint32_t* tInf) {
  uint32_t t0 = micros();
  feat::logmel(win, fvec);
  float sc = tin->params.scale; int zp = tin->params.zero_point;
  for (int i = 0; i < feat::FRAMES * feat::NMELS; i++) {
    int q = (int)lroundf(((fvec[i] - NORM_MEAN) / NORM_STD) / sc) + zp;
    tin->data.int8[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }
  uint32_t t1 = micros();
  interp->Invoke();
  uint32_t t2 = micros();
  int best = 0; float bp = -1;
  for (int c = 0; c < 4; c++) {
    float p = (tout->data.int8[c] - tout->params.zero_point) * tout->params.scale; lastProbs[c] = p;
    if (p > bp) { bp = p; best = c; }
  }
  *conf = bp; *tFeat = t1 - t0; *tInf = t2 - t1;
  return best;
}

// ---------------- OLED ----------------
void show(const char* l1, const char* l2, const char* l3, const char* l4) {
  oled.clearDisplay(); oled.setTextSize(1); oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0);  oled.print(l1);
  oled.setCursor(0, 16); oled.print(l2);
  oled.setCursor(0, 32); oled.print(l3);
  oled.setCursor(0, 48); oled.print(l4);
  oled.display();
}

// ---------------- web dashboard ----------------
String stamp(const Evt& e) {
  if (e.epoch > 1600000000) { struct tm t; localtime_r(&e.epoch, &t); char b[16]; strftime(b, sizeof(b), "%H:%M:%S", &t); return String(b); }
  return "t+" + String(e.startMs / 1000) + "s";
}
void handleRoot() { server.send_P(200, "text/html", DASH_HTML); }
void handleJson() {
  String j = "[";
  for (int i = 0; i < histN; i++) {
    const Evt& e = hist[i];
    j += String(i ? "," : "") + "{\"id\":" + String(e.id) + ",\"time\":\"" + stamp(e) + "\",\"source\":\"" + CLASS_NAME[e.cls] + "\",\"confidence\":" + String(e.conf, 2) +
         ",\"duration_s\":" + String(e.dur, 1) + ",\"priority\":\"" + CLASS_PRIO[e.cls] + "\"}";
  }
  server.sendHeader("Access-Control-Allow-Origin", "*"); server.sendHeader("Access-Control-Allow-Private-Network", "true");
  server.send(200, "application/json", j + "]");
}
void addEvent(const Evt& e) {  // newest first, keep 10
  for (int i = min(histN, 19); i > 0; i--) hist[i] = hist[i - 1];
  hist[0] = e; hist[0].id = ++evSeq; evCount[e.cls]++; if (histN < 20) histN++;
}

// ---------------- main ----------------
static int votes[VOTES]; static float vconf[VOTES]; static int vi = 0;
static bool active = false; static int below = 0; static uint32_t evStart = 0; static time_t evEpoch = 0;
static int confirmedCls = -1; static float sumConf[4]; static int cntConf[4];
static uint32_t lastProc = 0, nWin = 0, sumInf = 0, maxInf = 0, sumFeat = 0, maxFeat = 0;
static char lastLine[24] = "No event yet"; static uint32_t lastShowMs = 0;

void handleLive() {
  char b[420];
  snprintf(b, sizeof b, "{\"s\":%d,\"rms\":%.5f,\"on\":%.4f,\"off\":%.4f,\"cf\":%d,\"p\":[%.2f,%.2f,%.2f,%.2f],\"n\":%u,\"c\":[%u,%u,%u,%u],\"up\":%lu,\"ti\":%lu,\"tf\":%lu,\"win\":%lu,\"heap\":%u,\"th\":%.2f}",
           active ? 1 : 0, liveRms, RMS_ON, RMS_OFF, active ? confirmedCls : -1, lastProbs[0], lastProbs[1], lastProbs[2], lastProbs[3], evSeq,
           evCount[0], evCount[1], evCount[2], evCount[3], millis() / 1000, lastTi, lastTf, nWin, (unsigned)ESP.getFreeHeap(), CONF_THRESHOLD);
  server.sendHeader("Cache-Control", "no-store"); server.sendHeader("Access-Control-Allow-Origin", "*"); server.sendHeader("Access-Control-Allow-Private-Network", "true");
  server.send(200, "application/json", b);
}
void handleNotFound() {                      // answers the browser's CORS pre-check, 404 otherwise
  server.sendHeader("Access-Control-Allow-Origin", "*");
  if (server.method() == HTTP_OPTIONS) {
    server.sendHeader("Access-Control-Allow-Methods", "GET, OPTIONS"); server.sendHeader("Access-Control-Allow-Headers", "*");
    server.sendHeader("Access-Control-Allow-Private-Network", "true"); server.send(204);
  } else server.send(404, "text/plain", "Not found");
}
void resetVotes() { for (int i = 0; i < VOTES; i++) { votes[i] = -1; vconf[i] = 0; } vi = 0; confirmedCls = -1; for (int c = 0; c < 4; c++) { sumConf[c] = 0; cntConf[c] = 0; } }

void setup() {
  Serial.begin(115200); delay(500);
  Wire.begin(PIN_SDA, PIN_SCL);
  if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) Serial.println("OLED not found");
  show("PBEL Noise Monitor", "Starting...", "", "");
  ring = (int16_t*)heap_caps_calloc(RING, sizeof(int16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  if (!ring) { Serial.println("Ring buffer allocation FAILED"); for (;;) delay(1000); }
  if (!feat::init()) { Serial.println("Mel filterbank init FAILED"); for (;;) delay(1000); }
  Serial.printf("[boot] ring + features ok, free heap %u\n", (unsigned)ESP.getFreeHeap());
  if (NORM_MEAN == 0.0f && NORM_STD == 1.0f) { Serial.println("WARNING: NORM_MEAN/NORM_STD are still the placeholders - copy them from norm.json!"); show("SET NORM_MEAN/STD", "from norm.json", "", ""); delay(3000); }
  modelBegin();
  Serial.println("[boot] model ok");
  i2sBegin();
  Serial.println("[boot] I2S ok");
  xTaskCreatePinnedToCore(audioTask, "audio", 4096, NULL, 5, NULL, 1);   // core 1: Wi-Fi runs on core 0 and cannot disturb capture
  bool sta = false;
  if (strlen(WIFI_SSID)) {
    WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID, WIFI_PASS);
    for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) delay(500);
    sta = WiFi.status() == WL_CONNECTED;
  }
  if (sta) { configTime(19800, 0, "pool.ntp.org"); ipStr = WiFi.localIP().toString(); wifiOn = true; }
  else if (AP_FALLBACK) { WiFi.mode(WIFI_AP); WiFi.softAP(AP_SSID, AP_PASS); ipStr = WiFi.softAPIP().toString(); wifiOn = true; }
  if (wifiOn) {
    server.on("/", handleRoot); server.on("/api/live", handleLive); server.on("/api/events", handleJson); server.on("/events", handleJson); server.onNotFound(handleNotFound);
    server.begin(); Serial.print("Dashboard: http://"); Serial.println(ipStr);
  }
  Serial.printf("[boot] ready, free heap %u\n", (unsigned)ESP.getFreeHeap());
  resetVotes();
  lastProc = totalSamples;
  show("Listening...", wifiOn ? ipStr.c_str() : "No Wi-Fi", wifiOn ? (sta ? "Router Wi-Fi" : AP_SSID) : "", "");
}

void loop() {
  if (wifiOn) server.handleClient();
  uint32_t tot = totalSamples;
  if (tot - lastProc < HOP || tot < (uint32_t)feat::WIN) { delay(5); return; }
  lastProc += HOP;
  uint32_t end = lastProc;                       // process the window that ends at 'end'
  for (int i = 0; i < feat::WIN; i++) win[i] = ring[(end - feat::WIN + i) % RING];
  float blockRms = feat::rms16(win + feat::WIN - HOP, HOP); liveRms = blockRms;

  char l1[24], l2[24], l3[24];
  if (!active) {
    if (blockRms > RMS_ON) { active = true; below = 0; resetVotes(); evStart = millis() - 250; evEpoch = (time(nullptr) > 1600000000) ? time(nullptr) - 1 : 0; }
  } else {
    below = (blockRms < RMS_OFF) ? below + 1 : 0;
  }

  if (active) {
    float conf; uint32_t tf, ti;
    int cls = classify(&conf, &tf, &ti);
    lastTi = ti; lastTf = tf; nWin++; sumInf += ti; sumFeat += tf; if (ti > maxInf) maxInf = ti; if (tf > maxFeat) maxFeat = tf;
    int v = (conf >= CONF_THRESHOLD) ? cls : -1;               // -1 = UNKNOWN / LOW CONFIDENCE
    votes[vi] = v; vconf[vi] = conf; vi = (vi + 1) % VOTES;
    int cnt[4] = {0, 0, 0, 0}; for (int i = 0; i < VOTES; i++) if (votes[i] >= 0) cnt[votes[i]]++;
    int top = 0; for (int c = 1; c < 4; c++) if (cnt[c] > cnt[top]) top = c;
    if (cnt[top] >= NEED && top != 0) {
      if (confirmedCls != top) { confirmedCls = top; }
    }
    if (v > 0) { sumConf[v] += conf; cntConf[v]++; }
    Serial.printf("t=%lu rms=%.4f raw=%s conf=%.2f vote=%s feat=%lums inf=%lums\n", millis(), blockRms, CLASS_NAME[cls], conf,
                  v < 0 ? "UNKNOWN" : CLASS_NAME[v], tf / 1000, ti / 1000);
    snprintf(l1, sizeof l1, "EVENT  rms %.3f", blockRms);
    if (confirmedCls >= 0) { snprintf(l2, sizeof l2, "%s", CLASS_NAME[confirmedCls]); snprintf(l3, sizeof l3, "Conf %d%%", cntConf[confirmedCls] ? (int)(100 * sumConf[confirmedCls] / cntConf[confirmedCls]) : 0); }
    else { snprintf(l2, sizeof l2, "%s", cnt[top] >= NEED ? "Ambient" : "Analysing..."); snprintf(l3, sizeof l3, "%s", v < 0 ? "LOW CONFIDENCE" : ""); }
    show(l1, l2, l3, lastLine);

    if (below >= HANG_BLOCKS) {                                   // event ended
      active = false;
      if (confirmedCls >= 0) {
        Evt e; e.startMs = evStart; e.epoch = evEpoch; e.cls = confirmedCls;
        e.dur = (millis() - evStart - below * 250.0f) / 1000.0f; if (e.dur < 0.25f) e.dur = 0.25f;
        e.conf = cntConf[confirmedCls] ? sumConf[confirmedCls] / cntConf[confirmedCls] : 0;
        addEvent(e);
        snprintf(lastLine, sizeof lastLine, "%.9s %d%% %.1fs", CLASS_NAME[e.cls], (int)(e.conf * 100), e.dur);
        Serial.printf("EVENT CONFIRMED: Source: %s | Confidence: %d%% | Duration: %.1f s | Priority: %s\n", CLASS_NAME[e.cls], (int)(e.conf * 100), e.dur, CLASS_PRIO[e.cls]);
      }
      Serial.printf("STATS windows=%lu feat avg/max=%lu/%lu us  infer avg/max=%lu/%lu us  min free heap=%u\n", nWin, nWin ? sumFeat / nWin : 0, maxFeat,
                    nWin ? sumInf / nWin : 0, maxInf, (unsigned)ESP.getMinFreeHeap());
      show("Listening...", "", "Last event:", lastLine);
    }
  } else {
    if (millis() - lastShowMs > 500) { lastShowMs = millis(); snprintf(l1, sizeof l1, "Idle  rms %.3f", blockRms); show(l1, "", "Last event:", lastLine); Serial.printf("idle rms=%.4f", blockRms); if (DEBUG_RAW) Serial.printf("   raw24 min=%ld max=%ld", (long)rawMin, (long)rawMax); Serial.println(); }
  }
}
