from openalpr import Alpr
from argparse import ArgumentParser
import random

parser = ArgumentParser(description='OpenALPR Python Test Program')

parser.add_argument("-c", "--country", dest="country", action="store", default="us",
                    help="License plate Country")

parser.add_argument("--config", dest="config", action="store", default="conf.txt",
                    help="Path to openalpr.conf config file")

parser.add_argument("--runtime_data", dest="runtime_data", action="store", default="runtime_data",
                    help="Path to OpenALPR runtime_data directory")

parser.add_argument('--plate_image', help='License plate image file')

options = parser.parse_args()

alpr = None
try:
    alpr = Alpr(options.country, options.config, options.runtime_data)

    if not alpr.is_loaded():
        print("Error loading OpenALPR")
    else:
        print("Using OpenALPR " + alpr.get_version())

        alpr.set_top_n(7)
        alpr.set_default_region("wa")
        alpr.set_detect_region(False)
        jpeg_bytes = open(options.plate_image, "rb").read()
        loop = True
        while (loop):
            jpeg_bytes = bytes(jpeg_bytes)
            results = alpr.recognize_array(jpeg_bytes)
            jpeg_bytes = bytearray(jpeg_bytes)

            print(results['results'])

            if (len(results['results']) == 0):
                loop = False
            else:
                print(len(jpeg_bytes))
                for i in range(500, len(jpeg_bytes)):
                    sum = jpeg_bytes[i] + int(255*random.random())
                    jpeg_bytes[i] = sum % 255
                with open('output.jpg', 'wb') as f:
                    f.write(jpeg_bytes)


finally:
    if alpr:
        alpr.unload()
