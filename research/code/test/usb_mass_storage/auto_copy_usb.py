#import gobject
from dbus.mainloop.glib import DBusGMainLoop  
from gi.repository import GObject as gobject
import dbus
import dbus.service
# if getattr(dbus, 'version', (0,0,0)) >= (0,41,0):
#     import dbus.glib

  
#import gobject
import sys
import os

class DeviceManager:
    def __init__(self):
        self.bus = dbus.SystemBus()
        self.bus.add_signal_receiver(self.device_added,
                        'DeviceAdded',
                        'org.freedesktop.Hal.Manager',
                        'org.freedesktop.Hal',
                        '/org/freedesktop/Hal/Manager')

        self.bus.add_signal_receiver(self.device_removed,
                        'DeviceRemoved',
                        'org.freedesktop.Hal.Manager',
                        'org.freedesktop.Hal',
                        '/org/freedesktop/Hal/Manager')

    def udi_to_device(self, udi):
        return self.bus.get_object("org.freedesktop.Hal", udi)

    def device_added(self, udi):
        print ('Added', udi)
        properties = self.udi_to_device(udi).GetAllProperties()
        if properties.get('info.category') == u'volume':
            label, dev = properties.get('volume.label'), properties.get('block.device')
            print ('Mounting %s on /media/%s' %(dev, label))
            os.system('pmount %s /media/%s' %(dev, label))

    def device_removed(self, udi):
        print ('Removed', udi)

if __name__ == '__main__':
    m = DeviceManager()

    mainloop = DBusGMainLoop(set_as_default=True) #gobject.MainLoop()
    try:
        #gobject.mainloop.run()
        m.bus.SessionBus(mainloop=mainloop)
    except KeyboardInterrupt:
        gobject.mainloop.quit()
        print ('Exiting...')
        sys.exit(0)