#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>
#include <ESP8266WiFi.h>
RF24 radio(D3,D8); // CE, CSN
float indoor[3];

const byte address[6] = "00001";
void setup() {
  Serial.begin(9600);
  radio.begin();
  radio.openReadingPipe(0, address);
  radio.setPALevel(RF24_PA_MAX);
  radio.startListening();
}
void loop() {
  if (radio.available()) {
    radio.read(indoor, sizeof(indoor));
    Serial.println(indoor[0]);
    Serial.println(indoor[1]);
    Serial.println(indoor[2]);
  }
}