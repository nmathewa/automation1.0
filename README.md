# automation1.0
## NRF24l01

![](https://www.electronicwings.com/public/images/user_images/images/Arduino/nRF24L01/Frequency%20Channel.png)

The multi-receiver capacity of nRF24L01 is having up to 6 channels (pipes) of radio communication open in a receiving or read mode simultaneously. This takes the form of a hub receiver (PRX - primary receiver) and up to six transmitter nodes (PTX1 - PTX6 primary transmitters). In the above diagram, six reading (Data) pipes are opened in the primary receiver hub (PRX). Each PTX node links to one of these pipes to use both in transmitting and receiving (TX toward the hub being the primary direction of data flow, but the PTX nodes are RX capable as well).

![](https://how2electronics.com/wp-content/uploads/2019/04/NRF24L01-Working-Principles-of-Channels-and-Addresses.png)


## 20/12/2020
-----------

![display](images/display1.jpeg)
![display](images/display2.jpeg)

## 25/12/2020

1. Using LORA RFM69H module instead of the NRF24l01
   - NRF24l01 practical range is limited to 100 meters 
   - RFM69H has a range of 500 meter without directional antenna 
![](https://images-na.ssl-images-amazon.com/images/I/51QjC7kO-kL.jpg)

2. Outdoor unit will equipped with RFM while all indoor units are with nrf24l01
3. thinking about using raspberry pi zero instead of nodeMCU as a gateway
   - Server setup can be done on raspberry pi
   - more RAM and flash (memory card)
   - additional features
   - The network can be expanded in the future

![](https://www.raspberrypi.org/homepage-9df4b/static/1dfa03d09c1f3e446e8d936dfb92267f/ae23f/6b0defdbbf40792b64159ab8169d97162c380b2c_raspberry-pi-zero-1-1755x1080.jpg)


## 14/03/21

![](images/gateway.jpg)

![](images/node1.jpg)
