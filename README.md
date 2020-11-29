# automation1.0
## NRF24l01
The multi-receiver capacity of nRF24L01 is having up to 6 channels (pipes) of radio communication open in a receiving or read mode simultaneously. This takes the form of a hub receiver (PRX - primary receiver) and up to six transmitter nodes (PTX1 - PTX6 primary transmitters). In the above diagram, six reading (Data) pipes are opened in the primary receiver hub (PRX). Each PTX node links to one of these pipes to use both in transmitting and receiving (TX toward the hub being the primary direction of data flow, but the PTX nodes are RX capable as well).
