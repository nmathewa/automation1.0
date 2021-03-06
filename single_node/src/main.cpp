#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_PCD8544.h>
#include <Wire.h>

//#include "DHT.h"
#include <RF24.h>
//#define DHTPIN 10
//#define DHTTYPE DHT11
#include  "BME280I2C.h"
#include <Adafruit_BMP085.h>
//DHT dht(DHTPIN, DHTTYPE);
RF24 radio(8,9);



const byte address[6] = "00001";
//struct MyData {
//  byte h;
//  byte t;
//};

// Declare LCD object for software SPI
// Adafruit_PCD8544(CLK,DIN,D/C,CE,RST);
Adafruit_PCD8544 display = Adafruit_PCD8544(7, 6, 5, 4, 3);
//Adafruit_BMP085 bmp;
BME280I2C bme;   
int cd = 1 ;
int rotatetext = 1;
int in_delay = 10;
float indoor[3];
//MyData data;
void setup()   {
	Serial.begin(9600);
    bme.begin();
    //bmp.begin();
    radio.begin();
    radio.openWritingPipe(address);
    radio.setPALevel(RF24_PA_MAX);
    radio.stopListening();
	//Initialize Display
	display.begin();
	// you can change the contrast around to adapt the display for the best viewing!
	display.setContrast(57);
    //dht.begin();
	// Clear the buffer.
	display.clearDisplay();
    analogWrite(11,10);
    //radio.printDetails();
}

void loop() {
    //float temp = dht.readTemperature();
    float temp = bme.temp();
    //float hum = dht.readHumidity();
    float hum = bme.hum();
    //float pre = bmp.readPressure();
    float pre = bme.pres();
    //float hin = dht.computeHeatIndex(temp,hum,false);
    indoor[0] = temp;
    indoor[1] = hum;
    indoor[2] = pre;

    radio.write(indoor, sizeof(indoor));
    //float alt = bmp.readAltitude();
    //radio.write(&data, sizeof(MyData));
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
        display.print(pre);
        display.print("hPa");
	    //display.display();
        //Serial.print(pre);
        //delay(in_delay);
    }
    else {
        display.drawRect(0, 0, 84, 12,BLACK);
        display.setCursor(28,2);
        display.print("Indoor");
        display.setCursor(2,17);
        //display.print("Hindex");
        display.print("Altitude");
        display.setCursor(6,30);
        display.drawRect(0, 14, 41, 30,BLACK);
        display.drawRect(42, 14, 41, 30,BLACK);
        cd = 0;
        //delay(in_delay);
    }
    Serial.print("PRE: ");
    Serial.println(pre);
    Serial.print("HUM: ");
    Serial.println(hum);
    Serial.print("TEM: ");
    Serial.println(temp);
    display.display();
	delay(5000);

        	
}
