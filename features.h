// Log-Mel feature extraction - port of 1_prepare_dataset.py (NumPy), memory-lean version.
// Pure C++ (no Arduino dependencies) so it can be unit-tested on a PC.
// Input is 16-bit PCM (what the I2S task stores); sparse mel filterbank saves ~38 KB RAM.
#pragma once
#include <math.h>
#include <stdint.h>

namespace feat {
constexpr int SR = 16000, WIN = 16000, NFFT = 512, HOP = 256;
constexpr int NMELS = 40, NBINS = NFFT / 2 + 1, FRAMES = 1 + (WIN - NFFT) / HOP;   // 61
constexpr float TARGET_RMS = 0.05f;
constexpr double FMIN = 0.0, FMAX = 8000.0;
constexpr int MAXW = 700;                       // non-zero mel weights (each bin is in <= 2 filters)

static float fbw[MAXW];                         // packed filter weights
static uint16_t fbStart[NMELS], fbLen[NMELS], fbOff[NMELS];
static float hannw[NFFT];
static float twr[NFFT / 2], twi[NFFT / 2];
static uint16_t brev[NFFT];

static inline double hz2mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static inline double mel2hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

// returns false if the packed weights do not fit (should never happen)
static inline bool init() {
  double hz[NMELS + 2];
  double m0 = hz2mel(FMIN), m1 = hz2mel(FMAX);
  for (int i = 0; i < NMELS + 2; i++) hz[i] = mel2hz(m0 + (m1 - m0) * i / (NMELS + 1));
  int off = 0;
  for (int i = 0; i < NMELS; i++) {
    int first = -1, last = -1;
    fbOff[i] = (uint16_t)off;
    for (int k = 0; k < NBINS; k++) {
      double f = (double)k * SR / NFFT;
      double up = (f - hz[i]) / (hz[i + 1] - hz[i]), dn = (hz[i + 2] - f) / (hz[i + 2] - hz[i + 1]);
      double v = up < dn ? up : dn;
      if (v > 0) { if (first < 0) first = k; last = k; }
    }
    if (first < 0) { fbStart[i] = 0; fbLen[i] = 0; continue; }
    fbStart[i] = (uint16_t)first; fbLen[i] = (uint16_t)(last - first + 1);
    if (off + fbLen[i] > MAXW) return false;
    for (int k = first; k <= last; k++) {
      double f = (double)k * SR / NFFT;
      double up = (f - hz[i]) / (hz[i + 1] - hz[i]), dn = (hz[i + 2] - f) / (hz[i + 2] - hz[i + 1]);
      double v = up < dn ? up : dn;
      fbw[off++] = (float)(v > 0 ? v : 0);
    }
  }
  for (int n = 0; n < NFFT; n++) hannw[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * n / (NFFT - 1)));  // np.hanning
  for (int i = 0; i < NFFT / 2; i++) { twr[i] = (float)cos(-2.0 * M_PI * i / NFFT); twi[i] = (float)sin(-2.0 * M_PI * i / NFFT); }
  for (int i = 0; i < NFFT; i++) { int r = 0; for (int b = 0; b < 9; b++) if (i & (1 << b)) r |= 1 << (8 - b); brev[i] = (uint16_t)r; }
  return true;
}

static inline void fft512(float* re, float* im) {
  for (int i = 0; i < NFFT; i++) { int j = brev[i]; if (j > i) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (int len = 2; len <= NFFT; len <<= 1) {
    int half = len >> 1, step = NFFT / len;
    for (int i = 0; i < NFFT; i += len)
      for (int j = 0; j < half; j++) {
        float wr = twr[j * step], wi = twi[j * step];
        int u = i + j, v = u + half;
        float tr = re[v] * wr - im[v] * wi, ti = re[v] * wi + im[v] * wr;
        re[v] = re[u] - tr; im[v] = im[u] - ti; re[u] += tr; im[u] += ti;
      }
  }
}

// RMS of 16-bit samples, in full-scale units (1.0 = 32768)
static inline float rms16(const int16_t* x, int n) {
  double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
  return (float)sqrt(s / n / (32768.0 * 32768.0) + 1e-12);
}

// x: WIN int16 samples.  out: FRAMES*NMELS floats, time-major (out[t*NMELS+m]).
static inline void logmel(const int16_t* x, float* out) {
  float g = TARGET_RMS / rms16(x, WIN) / 32768.0f;
  static float re[NFFT], im[NFFT], pw[NBINS];
  for (int t = 0; t < FRAMES; t++) {
    const int16_t* p = x + t * HOP;
    for (int n = 0; n < NFFT; n++) { re[n] = (float)p[n] * g * hannw[n]; im[n] = 0.0f; }
    fft512(re, im);
    for (int k = 0; k < NBINS; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    for (int m = 0; m < NMELS; m++) {
      float s = 0; const float* w = fbw + fbOff[m]; const float* q = pw + fbStart[m];
      for (int k = 0; k < fbLen[m]; k++) s += w[k] * q[k];
      out[t * NMELS + m] = logf(s + 1e-6f);
    }
  }
}
}  // namespace feat
