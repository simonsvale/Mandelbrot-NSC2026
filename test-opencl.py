import pyopencl as cl 


if __name__ == "__main__":

    # Get the platforms
    platforms = cl.get_platforms()

    for platform in platforms:
        print("Platform:", platform.name)

        # Get devices for the platform.
        devices = platform.get_devices()

        # Print all devices for that platform.
        for device in devices:
            print("Device:", device.name)