import os
import ptvsd
import socket
import hspmd

def distributed_init():
    hostname = socket.gethostname()
    # os.environ['HSPMD_LOCAL_HOSTNAME'] = os.environ['HOSTNAME']
    os.environ['HSPMD_LOCAL_HOSTNAME'] = hostname

    hspmd.init_comm_group(8)
    local_device = hspmd.local_device()
    all_devices = hspmd.global_device_group()
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')
    # used for debug
    # ptvsd.enable_attach(address =('127.0.0.1', 4000 + all_devices.get_index(local_device)))
    # ptvsd.wait_for_attach()
    return local_device, all_devices