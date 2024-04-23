from openalpr import Alpr
from argparse import ArgumentParser
import random

parser = ArgumentParser(description='OpenALPR Python Test Program')

parser.add_argument("-c", "--country", dest="country", action="store", default="us",
                  help="License plate Country" )

parser.add_argument("--config", dest="config", action="store", default="conf.txt",
                  help="Path to openalpr.conf config file" )

parser.add_argument("--runtime_data", dest="runtime_data", action="store", default="runtime_data",
                  help="Path to OpenALPR runtime_data directory" )

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
        print(jpeg_bytes)
        loop = True
        while(loop):
            results = alpr.recognize_array(jpeg_bytes)

            if(len(results) == 0):
                loop = False
            else:
                patch = []
                for i in range(0,len(jpeg_bytes)):
                    patch += chr(256*random.random())
                jpeg_bytes += bytes(patch, 'utf-8')



finally:
    if alpr:
        alpr.unload()
