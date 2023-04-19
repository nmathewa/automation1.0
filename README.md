# automation1.0
## NRF24l01

<img src="https://www.electronicwings.com/public/images/user_images/images/Arduino/nRF24L01/Frequency%20Channel.png" width="200" height="200">

The multi-receiver capacity of nRF24L01 is having up to 6 channels (pipes) of radio communication open in a receiving or read mode simultaneously. This takes the form of a hub receiver (PRX - primary receiver) and up to six transmitter nodes (PTX1 - PTX6 primary transmitters). In the above diagram, six reading (Data) pipes are opened in the primary receiver hub (PRX). Each PTX node links to one of these pipes to use both in transmitting and receiving (TX toward the hub being the primary direction of data flow, but the PTX nodes are RX capable as well).



## 20/12/2020
-----------


<img src="images/display2.jpeg" width="400" height="200">

<img src="images/display1.jpeg" width="200" height="200">

## 25/12/2020

1. Using LORA RFM69H module instead of the NRF24l01
   - NRF24l01 practical range is limited to 100 meters 
   - RFM69H has a range of 500 meter without directional antenna 

<img src="https://images-na.ssl-images-amazon.com/images/I/51QjC7kO-kL.jpg" width="200" height="200">

1. Outdoor unit will equipped with RFM while all indoor units are with nrf24l01
2. thinking about using raspberry pi zero instead of nodeMCU as a gateway
   - Server setup can be done on raspberry pi
   - more RAM and flash (memory card)
   - additional features
   - The network can be expanded in the future

![](https://www.raspberrypi.org/homepage-9df4b/static/1dfa03d09c1f3e446e8d936dfb92267f/ae23f/6b0defdbbf40792b64159ab8169d97162c380b2c_raspberry-pi-zero-1-1755x1080.jpg)

<img src="https://cdn-shop.adafruit.com/1200x900/2885-06.jpg" width="250" height="200">

## 14/03/21

<img src="images/gateway.jpg" width="200" height="200">


<img src="images/node1.jpg" width="300" height="250">

