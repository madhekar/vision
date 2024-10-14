import glib
import gudev
import pynotify
import sys

def callback(client, action, device, user_data):
    device_vendor = device.get_property("ID_VENDOR_ENC")
    device_model = device.get_property("ID_MODEL_ENC")
    if action == "add":
        notify = pynotify.Notofication("usb device added", "%s : %s is now connected to the system." % (device_vendor, device_model))
        notify.show()
    elif action == "remove":
        notify = pynotify.Notofication("usb device removed", "%s : %s is now disconnected to the system." % (device_vendor, device_model))   
        notify.show()

if not pynotify.init("usb device notifier"):
        sys.exit("can not connect to notification daemon!")    

client = gudev.Client(["usb/usb_device"])
client.connect("uevent", callable, None)

loop = glib.MainLoop()
loop.run()