import time
from toolbox.IO.nidaq import DigitalInputU32


if __name__ == '__main__':
    

    di = DigitalInputU32(device='Dev2',
                         lines=24,
                         binary="C:/di_test",
                         clock_speed=100000.0,
                         buffer_size=10000)

    di.start()

    time.sleep(20)

    di.stop()

    di.clear()

    time.sleep(1)

