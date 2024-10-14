import usb.core

dev = usb.core.find(idVendor=0x0781, idProduct=0x5575)
ep = dev[0].interfaces()[0].endpoints()[0]
i=dev[0].interfaces()[0].bInterfaceNumber

print('device info:\n ', dev, '\n\n end point:\n ', ep, '\n\n interface number:\n ', i)
