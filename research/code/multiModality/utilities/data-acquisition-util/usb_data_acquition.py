import re
import subprocess

import dbus

"""
import dbus
bus = dbus.SystemBus()
udisks = bus.get_object("org.freedesktop.UDisks", "/org/freedesktop/UDisks")
udisks = dbus.Interface(udisks, 'org.freedesktop.UDisks')
devices = udisks.get_dbus_method('EnumerateDevices')()

 qdbus --system org.freedesktop.UDisks2
sudo apt install qdbus
sudo apt install qdbus-qt5
sudo apt install qtchooser

"""

def usbDevices():
    device_re = re.compile("Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$", re.I)
    df = subprocess.check_output("lsusb")

    print(df)

    devices = []
    for i in df.split('\n'):
        if i:
            info = device_re.match(i)
            if info:
                dinfo = info.groupdict()
                dinfo['device'] = '/dev/bus/usb/%s/%s' % (dinfo.pop('bus'), dinfo.pop('device'))
                devices.append(dinfo)
    print (devices)

def usbConnectedDevices():
    bus = dbus.SessionBus()

    ud_maanager_obj = bus.get_object('org.freedesktop.UDisks2', '/org/freedesktop/UDisks2')

    om = dbus.Interface(ud_maanager_obj, 'org.freedesktop.DBus.ObjectManager')

    for k, v in om.GetManagedObjects().iteritems():
        drived_info = v.get('org.freedesktop.UDisks2', {})
        if drived_info.get('ConnectedBus') =='usb' and drived_info.get('Removable'):
            print('Device label: %s' % drived_info['id'])

def usbConnectedTest():
    bus = dbus.SystemBus()
    udisks = bus.get_object("org.freedesktop.UDisks", "/org/freedesktop/UDisks")
    udisks = dbus.Interface(udisks, "org.freedesktop.UDisks")
    devices = udisks.get_dbus_method("EnumerateDevices")()     
    print(devices)       

def diskConnected():
    bus = dbus.SystemBus()
    obj = bus.get_object("org.freedesktop.UDisks2", "/org/freedesktop/UDisks2")
    iface = dbus.Interface(obj, "org.freedesktop.DBus.ObjectManager")
    print(iface.bus_name, ':', iface)

    for k in iface.GetManagedObjects():
        print(k)


def GetProperties():
    bus = dbus.SystemBus()
    obj = bus.get_object(
        "org.freedesktop.UDisks2",
        "/org/freedesktop/UDisks2/drives/SanDisk_Cruzer_Glide_4C530001030505122305",
    )
    iface = dbus.Interface(
        obj, "org.freedesktop.DBus.Properties"
    )  # Here we use this 'magic' interface
    d = iface.GetAll("org.freedesktop.UDisks2.Drive")

    print(d)

    print(iface.Get("org.freedesktop.UDisks2.Drive", "Id"), iface.Get("org.freedesktop.UDisks2.Drive", "Vendor"))




diskConnected()

GetProperties()