// INMP441 microphone test - ESP32-S3, Arduino core 3.x
// Wiring: SCK->GPIO4, WS->GPIO5, SD->GPIO6, L/R->GND, VDD->3V3, GND->GND
// Tools: Board "ESP32S3 Dev Module", USB CDC On Boot "Enabled", Serial Monitor 115200.
#include "driver/i2s_std.h"

const int PIN_BCLK = 4, PIN_WS = 5, PIN_SD = 6;
i2s_chan_handle_t rx;

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nINMP441 mic test");
  i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  esp_err_t e1 = i2s_new_channel(&cc, NULL, &rx);
  i2s_std_config_t sc = {
    .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = { .mclk = I2S_GPIO_UNUSED, .bclk = (gpio_num_t)PIN_BCLK, .ws = (gpio_num_t)PIN_WS,
                  .dout = I2S_GPIO_UNUSED, .din = (gpio_num_t)PIN_SD, .invert_flags = {} },
  };
  esp_err_t e2 = i2s_channel_init_std_mode(rx, &sc);
  esp_err_t e3 = i2s_channel_enable(rx);
  Serial.printf("I2S setup codes (all must be 0): %d %d %d\n", (int)e1, (int)e2, (int)e3);
  Serial.println("Clap or talk close to the mic and watch LEFT / RIGHT swing.\n");
}

void loop() {
  static int32_t buf[1024];
  size_t got = 0;
  esp_err_t er = i2s_channel_read(rx, buf, sizeof(buf), &got, 1000);
  int n = got / 8;                                   // stereo frames
  long lmn = 0, lmx = 0, rmn = 0, rmx = 0;
  for (int i = 0; i < n; i++) {
    long l = buf[2 * i] >> 8, r = buf[2 * i + 1] >> 8;   // 24-bit values
    if (i == 0) { lmn = lmx = l; rmn = rmx = r; }
    if (l < lmn) lmn = l; if (l > lmx) lmx = l;
    if (r < rmn) rmn = r; if (r > rmx) rmx = r;
  }
  long lp = lmx - lmn, rp = rmx - rmn;               // peak-to-peak
  Serial.printf("read=%d frames=%d | LEFT %ld..%ld (p-p %ld) | RIGHT %ld..%ld (p-p %ld) -> ",
                (int)er, n, lmn, lmx, lp, rmn, rmx, rp);
  if (er != ESP_OK || n == 0)        Serial.println("NO DATA FROM I2S (clock problem)");
  else if (lp == 0 && rp == 0)       Serial.println("ALL ZERO: mic not sending (check SD, power, solder)");
  else if (lp > rp * 4 + 50)         Serial.println("MIC WORKS on LEFT channel");
  else if (rp > lp * 4 + 50)         Serial.println("MIC WORKS on RIGHT channel (set MIC_RIGHT_SLOT = true)");
  else                               Serial.println("Signal on both/noise - check L/R pin is firmly on GND");
  delay(500);
}
