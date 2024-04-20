# 569S OpenALPR

## Driver Instruction
Download openalpr and run:
```
sudo apt install -y libopencv-dev libtesseract-dev git cmake build-essential libleptonica-dev liblog4cplus-dev libcurl3-dev
sudo apt install beanstalkd
git clone https://github.com/openalpr/openalpr.git
cd openalpr/src
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_INSTALL_SYSCONFDIR:PATH=/etc ..
make
sudo make install
sudo python3 bindings/python/setup.py install
```
Now you are all set with environment:
```
sicheng@Sicheng:~/569S/open-alpr-ae$ python alpr_driver.py --plate_image dataset2/images/Cars6.png
Using OpenALPR 2.3.0
Plate #1
          Plate   Confidence
  -        0211N   90.748558
  -       0211NI   87.496651
  -        021IN   80.058990
  -        O211N   79.855385
  -        Q211N   79.139091
  -        02I1N   79.027061
  -        0Z11N   77.920334
```
