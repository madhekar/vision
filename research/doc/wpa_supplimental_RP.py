setup WiFi on a Raspberry Pi without needing to connect a monitor or keyboard to the Pi. Although you can configure WiFi using the graphical utility within the Raspbian Desktop this requires that you connect a keyboard, mouse and monitor to your Pi. It is sometimes useful to be able to do it before you’ve booted the Pi. This is especially useful when using the Pi Zero W or A+ models where attaching a keyboard and mouse requires a USB hub.

The following technique will allow you to take a fresh SD card, setup WiFi and boot a Pi without any other wires than a power cable. It should work for all Pi models but it’s easier on the devices with on-board Wi-Fi as you don’t need to worry about a WiFi dongle.

Step 1 – Create a fresh SD card using Raspbian image
Create fresh SD card using the latest available Raspbian image from the Official Download page.

NOTE: This method to setup WiFi must be completed before you boot this card for the first time. This is the point at which the system checks for the wpa_supplicant.conf file. If you have already booted the card you will need to re-write with a fresh image and continue.

Step 2 – Create a blank text file
Create a blank text file named “wpa_supplicant.conf”. Use a plain text editor rather than a Word Processor.

If using Windows you need to make sure the text file uses Linux/Unix style line breaks. I use Notepad++ (it’s free!) and this is easy to do using “Edit” > “EOL Conversion” > “UNIX/OSX Format”. “UNIX” is then shown in the status bar.

Notepad Plus Plus EOL Format

Insert the following content into the text file :

country=us
update_config=1
ctrl_interface=/var/run/wpa_supplicant

network={
 scan_ssid=1
 ssid="MyNetworkSSID"
 psk="Pa55w0rd1234"
}
Double check the SSID and password. Both the SSID and password should be surrounded by quotes.

The Country Code should be set the ISO/IEC alpha2 code for the country in which you are using your Pi. Common codes include :

gb (United Kingdom)
fr (France)
de (Germany)
us (United States)
se (Sweden)
Step 3 – Copy to SD Card
Copy the file to the boot partition on your SD card. In Windows this is the only partition you will be able to see. It will already contain some of the following files :

bootcode.bin
loader.bin
start.elf
kernel.img
cmdline.txt
Step 4 – Eject, Insert and Boot
Safely remove the SD card from your PC and insert into the Pi. Power up the Pi and once it has booted you should be connected to your WiFi network.

You may be able to use your Router admin interface to list connected devices. Your Pi should appear in the list with an assigned IP address.

Additional Thoughts
As Sebastian Bjurbom points out in the comments below you might want to take this opportunity to enable SSH as well. It is disabled by default but it is easy to enable by copying a blank text file named “ssh” to the boot partition. This can be done at the same time “wpa_supplicant.conf” is copied across.

If you save a copy of the wpa_supplicant.conf file (in a secure location) you can quickly copy it to all your SD cards when they have a fresh image written to them.

Troubleshooting
If after waiting a few minutes your Pi is not connected to your WiFi consider the following points :

Did you complete this method before booting this SD card for the very first time? If not start again from Step 1
Check wpa_supplicant.conf exists in the boot partition and the filename is correctly spelt
Check the file contains the text listed above
Double check every character in the SSID
Double check every character the password
Check the SSID and password are correctly surrounded with double quotes “….”
Ensure your text editor is using Linux style line breaks
