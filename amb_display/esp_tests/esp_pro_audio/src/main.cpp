/*
 * ambviz strip controller -- receives pixel data over UDP and drives a WS2812 strip.
 *
 * Wire format, one 4-byte record per pixel, several records per datagram:
 *
 *     | index | r | g | b |
 *
 * Only changed pixels are sent, so state persists between datagrams. This is
 * the protocol the `ambviz` Python package speaks; see amb_display/ambviz.
 *
 * Configuration lives in platformio.ini build flags and include/secrets.h --
 * nothing here should need editing to move the strip to another network or
 * change its length.
 *
 * Derived from work by Joey Babcock and Scott Lawson
 * (github.com/scottlawsonbc/audio-reactive-led-strip).
 */
#include <NeoPixelBus.h>

#if defined(ESP8266)
  #include <ESP8266WiFi.h>
  #include <ESP8266mDNS.h>
  #include <WiFiUdp.h>
#elif defined(ESP32)
  #include <WiFi.h>
  #include <ESPmDNS.h>
  #include <WiFiUdp.h>
#else
  #error "This firmware targets the ESP8266 or ESP32."
#endif

#include "secrets.h"

// ── build-time configuration (override in platformio.ini) ────────────────────
#ifndef LED_COUNT
  #define LED_COUNT 60
#endif
#ifndef LED_PIN
  #define LED_PIN 3          // ESP8266 DMA output is fixed to GPIO3 (RX)
#endif
#ifndef UDP_PORT
  #define UDP_PORT 7777
#endif
#ifndef MDNS_NAME
  #define MDNS_NAME "ambviz" // reachable as ambviz.local, so no static IP
#endif
#ifndef PRINT_FPS
  #define PRINT_FPS 1
#endif
#ifndef SERIAL_BAUD
  #define SERIAL_BAUD 115200
#endif

// A datagram holds at most 126 records (504 bytes) in the reference sender;
// 1024 leaves room without risking a stack-hungry buffer.
#define BUFFER_LEN 1024
#define RECORD_LEN 4

// The strip index arrives as one byte, so 255 pixels is the protocol ceiling.
static_assert(LED_COUNT > 0 && LED_COUNT <= 255,
              "LED_COUNT must be between 1 and 255: the protocol sends the pixel "
              "index as a single byte.");

WiFiUDP udp;
// Unsigned: the original read indices into a signed char, so anything past 127
// wrapped negative and wrote outside the strip.
uint8_t packetBuffer[BUFFER_LEN];
NeoPixelBus<NeoGrbFeature, Neo800KbpsMethod> ledstrip(LED_COUNT, LED_PIN);

uint32_t droppedRecords = 0;   // records naming a pixel this strip does not have

#if PRINT_FPS
  uint16_t packetCounter = 0;
  uint32_t secondTimer = 0;
#endif

static void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.hostname(MDNS_NAME);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.printf("\nconnecting to %s", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nconnected: %s\n", WiFi.localIP().toString().c_str());

  // DHCP plus mDNS, so the address is never duplicated in a config file.
  if (MDNS.begin(MDNS_NAME)) {
    MDNS.addService("ambviz", "udp", UDP_PORT);
    Serial.printf("also reachable as %s.local\n", MDNS_NAME);
  } else {
    Serial.println("mDNS failed to start; use the IP above");
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);
  Serial.printf("\nambviz strip: %d pixels on pin %d, udp %d\n",
                LED_COUNT, LED_PIN, UDP_PORT);

  ledstrip.Begin();
  ledstrip.Show();   // clear the strip

  connectWifi();
  udp.begin(UDP_PORT);
}

void loop() {
#if defined(ESP8266)
  MDNS.update();
#endif

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("wifi lost; reconnecting");
    connectWifi();
    udp.begin(UDP_PORT);
  }

  const int packetSize = udp.parsePacket();
  if (packetSize) {
    const int len = udp.read(packetBuffer, BUFFER_LEN);
    // Trailing bytes that do not form a whole record are ignored rather than
    // read past the end of the datagram.
    const int records = len / RECORD_LEN;
    for (int r = 0; r < records; r++) {
      const uint8_t *rec = &packetBuffer[r * RECORD_LEN];
      const uint8_t index = rec[0];
      if (index >= LED_COUNT) {
        droppedRecords++;    // sender thinks the strip is longer than it is
        continue;
      }
      ledstrip.SetPixelColor(index, RgbColor(rec[1], rec[2], rec[3]));
    }
    ledstrip.Show();
#if PRINT_FPS
    packetCounter++;
#endif
  }

#if PRINT_FPS
  if (millis() - secondTimer >= 1000U) {
    secondTimer = millis();
    Serial.printf("%u packets/s", packetCounter);
    if (droppedRecords) {
      Serial.printf("  (%u records dropped: index >= %d)", droppedRecords, LED_COUNT);
    }
    Serial.println();
    packetCounter = 0;
  }
#endif
}
