#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_PCD8544.h>
#include <Wire.h>
#include "DHT.h"
#define DHTPIN 10
#define DHTTYPE DHT11
#include <Adafruit_BMP085.h>
DHT dht(DHTPIN, DHTTYPE);

// Declare LCD object for software SPI
// Adafruit_PCD8544(CLK,DIN,D/C,CE,RST);
Adafruit_PCD8544 display = Adafruit_PCD8544(7, 6, 5, 4, 3);
Adafruit_BMP085 bmp;
int cd = 1 ;
int rotatetext = 1;
int in_delay = 10;

void setup()   {
	Serial.begin(9600);
    bmp.begin();
	//Initialize Display
	display.begin();
    
	// you can change the contrast around to adapt the display for the best viewing!
	display.setContrast(57);
    dht.begin();
	// Clear the buffer.
	display.clearDisplay();
    analogWrite(11,10);
}

void loop() {
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    float pre = bmp.readPressure();
    float hin = dht.computeHeatIndex(temp,hum,false);
    //float alt = bmp.readAltitude();
    
    display.clearDisplay();
    // Display Inverted Text
	display.setTextColor(BLACK); // 'inverted' text
    if (cd == 0){
        display.drawRect(0, 0, 84, 12,BLACK);
        display.setCursor(35,2);
        display.print("NMA");
        cd = cd+1;
        display.setCursor(2,17);
        display.drawRect(0, 14, 41, 15,BLACK);
    //display.print("T:");
	    display.print(temp);
        display.print('C');
        display.drawRect(42, 14, 41, 15,BLACK);
        display.setCursor(45,17);
	    display.print(hum);
        display.print('%');
        display.drawRect(0, 31, 84, 15,BLACK);
        display.setCursor(2,35);
        display.print(pre/100);
        display.print("hPa");
	    //display.display();
        Serial.print(pre);
        //delay(in_delay);
    }
    else {
        display.drawRect(0, 0, 84, 12,BLACK);
        display.setCursor(28,2);
        display.print("Indoor");
        display.setCursor(2,17);
        display.print("Hindex");
        display.setCursor(6,30);
        display.print(hin);
        display.drawRect(0, 14, 41, 30,BLACK);
        display.drawRect(42, 14, 41, 30,BLACK);
        cd = 0;
        //delay(in_delay);
    }
    display.display();
	delay(5000);
        	
}