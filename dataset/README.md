## The classes within the dataset are :

- laptop 
- keyboard 
- mouse 
- monitor
- phone 
- earbuds
- headphones 
- charger 
- usb cable 
- speaker 
- camera 
- controller 
- wallet 
- key
- backpack  
- watch 
- glasses 
- notebook 
- pen
- eraser 
- ruler
- calculator 
- paper 
- chair 
- cup
- bottle
- can
- plant
- spoon 
- chopsticks



To train your own model with a dataset of images, please carefully read and change the following variables:

- **IMAGES_PER_CLASS :** 
	- 40-60 if you have a relatively slow GPU, and anything between 600-1200 would give you a high quality model *default value is 120* 
*It is recommended that you refer to your computer specifications, and check if your GPU would be able to train on the amount of images you have before settign the value to that much*

- **Classes**
	- Add any new classes or remove any of the mentioned classes from the list given in the python file *(if you wish to train it for a specific object, while don't exactly need a different one)*

- **Web Scraping**
	- While there is a safe and useable script proivided, within this repo for a more manual route of data collection, It is highly recommended to use [OIDv4](https://github.com/EscVM/OIDv4_ToolKit), for downloading images via [OpenImages](https://storage.googleapis.com/openimages/web/index.html), as this provides open sourced images as well as labels for those particular images in a preset directory structure. In order to use OIDv4, please refer to the official GitHub repository for the Project and follow the instructions provided.
 
