from __future__ import print_function

import enum
import re
import struct
import sys
import threading
import time
import math
import pywt
import itertools
import pandas as pd
import numpy as np
import pickle
import string
import serial
import datetime

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import svm, grid_search
from sklearn import cross_validation

from serial.tools.list_ports import comports
from numpy import array
from common import *
# print(sys.modules.keys())
new_emg = []
process = []
filecount = 0
reList=[]
lightList=[]
heaviestList=[]
myPredictions =  {}
myPredictions['Worn: Rest'] = 0
myPredictions['Worn: Light'] = 0
myPredictions['Worn: Heavy'] = 0

def multichr(ords):
    if sys.version_info[0] >= 3:
        return bytes(ords)
    else:
        return ''.join(map(chr, ords))

def multiord(b):
    if sys.version_info[0] >= 3:
        return list(b)
    else:
        return map(ord, b)

def wavelet(emg):
    coeffs = pywt.wavedec(emg, 'db1', level= 3)
    cA3,cD3,cD2, cD1 = coeffs
    emg_cA=cA3
    emg_cD=cD3
    return(emg_cA,emg_cD)

# def classify(df5):
#     with open("s_data.csv", 'a') as f:
#         df5.to_csv(f, header=False)

# def modelFit():
#     seed = 40
#     df = pd.read_csv("s_data.csv")
#     X = df.loc[:, df.columns != "0c"]
#     X = X.loc[:, X.columns != "57c"] #label name dropped
#     y = df["58c"] #labels seperated
#     X = X.loc[:, X.columns != '58c']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,random_state=seed)
#     scaler = preprocessing.StandardScaler().fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test  = scaler.transform(X_test)
#     # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#     clf = svm.SVC(kernel='linear',C=2.0,gamma=1)
#     clf.fit(X_train, y_train)
#     predict = clf.predict(X_test)
#     y_pred = predict
#     y_true = y_test
#     print(accuracy_score(y_true, y_pred))
#     if accuracy_score > 70.0:
#         filename = 'model.sav'
#         pickle.dump(clf, open(filename, 'wb'))
#     #pickle.dump(clf,open("myo_train.pkl",'wb'))

def prediction(df5):
    #print(df5)
    #print(np.shape(df5))
    clf = pickle.load(open("./model/model.sav", 'rb'))
    predict= clf.predict(df5)
    y_pred = predict
    #print(y_pred)
    r=[]
    l=[]
    h=[]
    if y_pred == 0:
        # print("rest")
        return "Worn: Rest"
    if y_pred == 1:
        # print("lightest")
        # l.append('lightest')
        return "Worn: Light"
    if y_pred == 2:
        # h.append('heaviest')
        return "Worn: Heavy"


def preprocess(emgstr):
    emgstr = str(emgstr)
    df =  pd.DataFrame(columns=['emg1','emg2','emg3','emg4','emg5','emg6','emg7','emg8'])
    for x in "[]":
        emgstr = emgstr.replace(x,"")
    emgstr = emgstr.replace("), ","\n")
    for x in "()":
        emgstr = emgstr.replace(x,"")
    #print(emgstr)
    for x in emgstr.split('\n'):
            d = x
            processed = (d[:len(d[:-2])]+d[len(d[:-1]):].replace(' ',',')).replace(' ','')
            processed = processed.split(",")
            process = list(map(int,processed))
            #print(process)
            df.loc[len(df)] = process
    #print(df)
    return df

class Arm(enum.Enum):
    UNKNOWN = 0
    RIGHT = 1
    LEFT = 2


class XDirection(enum.Enum):
    UNKNOWN = 0
    X_TOWARD_WRIST = 1
    X_TOWARD_ELBOW = 2


class Pose(enum.Enum):
    REST = 0
    FIST = 1
    WAVE_IN = 2
    WAVE_OUT = 3
    FINGERS_SPREAD = 4
    THUMB_TO_PINKY = 5
    UNKNOWN = 255


class Packet(object):
    def __init__(self, ords):
        self.typ = ords[0]
        self.cls = ords[2]
        self.cmd = ords[3]
        self.payload = multichr(ords[4:])

    def __repr__(self):

        a = self.typ
        b = self.cls
        c = self.cmd
        d = multiord(self.payload)
        s = "{},{},{},{}\n".format(a,b,c,d)
        #fp.write(s)

        #return 'Packet(%02X, %02X, %02X, [%s])' % \
        #    (self.typ, self.cls, self.cmd,
        #     ' '.join('%02X' % b for b in multiord(self.payload)))


class BT(object):
    '''Implements the non-Myo-specific details of the Bluetooth protocol.'''
    def __init__(self, tty):
        self.ser = serial.Serial(port=tty, baudrate=9600, dsrdtr=1)
        self.buf = []
        self.lock = threading.Lock()
        self.handlers = []

    # internal data-handling methods
    def recv_packet(self, timeout=None):
        t0 = time.time()
        self.ser.timeout = None
        while timeout is None or time.time() < t0 + timeout:
            if timeout is not None:
                self.ser.timeout = t0 + timeout - time.time()
            c = self.ser.read()
            if not c:
                return None

            ret = self.proc_byte(ord(c))
            if ret:
                if ret.typ == 0x80:
                    self.handle_event(ret)
                return ret



    def recv_packets(self, timeout=.5):
    	        res = []
    	        t0 = time.time()
    	        while time.time() < t0 + timeout:
    	            p = self.recv_packet(t0 + timeout - time.time())
    	            if not p:
    	                return res
    	            res.append(p)
    	        return res

    def proc_byte(self, c):
        if not self.buf:
            if c in [0x00, 0x80, 0x08, 0x88]:  # [BLE response pkt, BLE event pkt, wifi response pkt, wifi event pkt]
                self.buf.append(c)
            return None
        elif len(self.buf) == 1:
            self.buf.append(c)
            self.packet_len = 4 + (self.buf[0] & 0x07) + self.buf[1]
            return None
        else:
            self.buf.append(c)

        if self.packet_len and len(self.buf) == self.packet_len:
            p = Packet(self.buf)
            self.buf = []
            return p
        return None

    def handle_event(self, p):
        for h in self.handlers:
            h(p)

    def add_handler(self, h):
        self.handlers.append(h)

    def remove_handler(self, h):
        try:
            self.handlers.remove(h)
        except ValueError:
            pass

    def wait_event(self, cls, cmd):
        res = [None]

        def h(p):
            if p.cls == cls and p.cmd == cmd:
                res[0] = p
        self.add_handler(h)
        while res[0] is None:
            self.recv_packet()
        self.remove_handler(h)
        return res[0]

    # specific BLE commands
    def connect(self, addr):

        return self.send_command(6, 3, pack('6sBHHHH', multichr(addr), 0, 6, 6, 64, 0))

    def get_connections(self):
        return self.send_command(0, 6)

    def discover(self):
        return self.send_command(6, 2, b'\x01')

    def end_scan(self):
        return self.send_command(6, 4)

    def disconnect(self, h):
        # print self.send_command(3, 0, pack('B', h))
        return self.send_command(3, 0, pack('B', h))

    def read_attr(self, con, attr):
        self.send_command(4, 4, pack('BH', con, attr))
        return self.wait_event(4, 5)

    def write_attr(self, con, attr, val):
        self.send_command(4, 5, pack('BHB', con, attr, len(val)) + val)
        return self.wait_event(4, 1)

    def send_command(self, cls, cmd, payload=b'', wait_resp=True):
        s = pack('4B', 0, len(payload), cls, cmd) + payload
        self.ser.write(s)

        while True:
            p = self.recv_packet()
            # no timeout, so p won't be None
            if p.typ == 0:
                return p
            # not a response: must be an event
            self.handle_event(p)


class MyoRaw(object):
    '''Implements the Myo-specific communication protocol.'''

    def __init__(self, tty=None):
        if tty is None:
            tty = self.detect_tty()
        if tty is None:
            raise ValueError('Myo dongle not found!')

        self.bt = BT(tty)
        self.conn = None
        self.emg_handlers = []
        self.currentSlide = []
        self.imu_handlers = []
        self.arm_handlers = []
        self.pose_handlers = []
        self.battery_handlers = []

    def detect_tty(self):
        for p in comports():
            if re.search(r'PID=2458:0*1', p[2]):
                # print('using device:', p[0])
                return p[0]

        return None

    def run(self, timeout=None):
        #print(self.bt.recv_packet(timeout))
        self.bt.recv_packet(timeout)

    def connect(self):
        # stop everything from before
        self.bt.end_scan()
        self.bt.disconnect(0)
        self.bt.disconnect(1)
        self.bt.disconnect(2)
        # start scanning
        print('scanning...')
        # if datetime

        count=0
        self.bt.discover()

        while True:
            count=count+1
            # print (count)
            p = self.bt.recv_packet()
            #print('scan response:', p)
            if count >10:
                print ("Not Worn")
                flag=True
                break
            if p.payload.endswith(b'\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47\xB9\xDE\x04\xA9\x01\x00\x06\xD5'):
                addr = list(multiord(p.payload[2:8]))
                # print (addr)
                break
        try:
            self.bt.end_scan()

            # connect and wait for status event
            conn_pkt = self.bt.connect(addr)
            self.conn = multiord(conn_pkt.payload)[-1]
            self.bt.wait_event(3, 0)

            # get firmware version
            fw = self.read_attr(0x17)
            _, _, _, _, v0, v1, v2, v3 = unpack('BHBBHHHH', fw.payload)
            # print('firmware version: %d.%d.%d.%d' % (v0, v1, v2, v3))

            self.old = (v0 == 0)

            if self.old:
                # don't know what these do; Myo Connect sends them, though we get data
                # fine without them
                self.write_attr(0x19, b'\x01\x02\x00\x00')
                # Subscribe for notifications from 4 EMG data channels
                self.write_attr(0x2f, b'\x01\x00')
                self.write_attr(0x2c, b'\x01\x00')
                self.write_attr(0x32, b'\x01\x00')
                self.write_attr(0x35, b'\x01\x00')

                # enable EMG data
                self.write_attr(0x28, b'\x01\x00')
                # enable IMU data
                self.write_attr(0x1d, b'\x01\x00')

                # Sampling rate of the underlying EMG sensor, capped to 1000. If it's
                # less than 1000, emg_hz is correct. If it is greater, the actual
                # framerate starts dropping inversely. Also, if this is much less than
                # 1000, EMG data becomes slower to respond to changes. In conclusion,
                # 1000 is probably a good value.
                C = 1000
                emg_hz = 50
                # strength of low-pass filtering of EMG data
                emg_smooth = 100

                imu_hz = 50


                # send sensor parameters, or we don't get any data
                self.write_attr(0x19, pack('BBBBHBBBBB', 2, 9, 2, 1, C, emg_smooth, C // emg_hz, imu_hz, 0, 0))

            else:
                name = self.read_attr(0x03)
                # print('device name: %s' % name.payload)

                # enable IMU data
                # self.write_attr(0x1d, b'\x01\x00')
                # # enable on/off arm notifications
                # self.write_attr(0x24, b'\x02\x00')
                # enable EMG notifications
                self.start_raw()
                # enable battery notifications
                self.write_attr(0x12, b'\x01\x10')
        except:
            # sys.exit()

            # self.disconnect()
            self.connect()


            # print ("Exception")

        # add data handlers
        def handle_data(p):
            if (p.cls, p.cmd) != (4, 5):
                return

            c, attr, typ = unpack('BHB', p.payload[:4])
            pay = p.payload[5:]

            if attr == 0x27:
                # Unpack a 17 byte array, first 16 are 8 unsigned shorts, last one an unsigned char
                vals = unpack('8HB', pay)
                # not entirely sure what the last byte is, but it's a bitmask that
                # seems to indicate which sensors think they're being moved around or
                # something
                emg = vals[:8]
                moving = vals[8]
                self.on_emg(emg, moving)
            # Read notification handles corresponding to the for EMG characteristics
            elif attr == 0x2b or attr == 0x2e or attr == 0x31 or attr == 0x34:
                '''According to http://developerblog.myo.com/myocraft-emg-in-the-bluetooth-protocol/
                each characteristic sends two secuential readings in each update,
                so the received payload is split in two samples. According to the
                Myo BLE specification, the data type of the EMG samples is int8_t.
                '''
                emg1 = struct.unpack('<8b', pay[:8])
                emg2 = struct.unpack('<8b', pay[8:])
                self.on_emg(emg1, 0)
                self.on_emg(emg2, 0)
            # Read IMU characteristic handle
            elif attr == 0x1c:
                vals = unpack('10h', pay)
                quat = vals[:4]
                acc = vals[4:7]
                gyro = vals[7:10]
                #imuvals=("{} {} {}\n".format(quat, acc, gyro))    #writing imu values to a file
                #fp2.write(imuvals)
                self.on_imu(quat, acc, gyro)
            # Read classifier characteristic handle
            elif attr == 0x23:
                typ, val, xdir, _, _, _ = unpack('6B', pay)

                if typ == 1:  # on arm
                    self.on_arm(Arm(val), XDirection(xdir))
                elif typ == 2:  # removed from arm
                    self.on_arm(Arm.UNKNOWN, XDirection.UNKNOWN)
                elif typ == 3:  # pose
                    self.on_pose(Pose(val))
            # Read battery characteristic handle
            elif attr == 0x11:
                battery_level = ord(pay)
                self.on_battery(battery_level)
            else:
                pass
                # print('data with unknown attr: %02X %s' % (attr, p))

        self.bt.add_handler(handle_data)

    def write_attr(self, attr, val):
        if self.conn is not None:
            self.bt.write_attr(self.conn, attr, val)

    def read_attr(self, attr):
        if self.conn is not None:
            return self.bt.read_attr(self.conn, attr)
        return None

    def disconnect(self):
        if self.conn is not None:
            self.bt.disconnect(self.conn)

    def sleep_mode(self, mode):
        # print("sleep")
        self.write_attr(0x19, pack('3B', 9, 1, mode))
        # print ("Rest")
        # self.bt.disconnect(self.conn)

    def power_off(self):
        self.write_attr(0x19, b'\x04\x00')

    def start_raw(self):

        ''' To get raw EMG signals, we subscribe to the four EMG notification
        characteristics by writing a 0x0100 command to the corresponding handles.
        '''
        self.write_attr(0x2c, b'\x01\x00')  # Suscribe to EmgData0Characteristic
        self.write_attr(0x2f, b'\x01\x00')  # Suscribe to EmgData1Characteristic
        self.write_attr(0x32, b'\x01\x00')  # Suscribe to EmgData2Characteristic
        self.write_attr(0x35, b'\x01\x00')  # Suscribe to EmgData3Characteristic

        '''Bytes sent to handle 0x19 (command characteristic) have the following
        format: [command, payload_size, EMG mode, IMU mode, classifier mode]
        According to the Myo BLE specification, the commands are:
            0x01 -> set EMG and IMU
            0x03 -> 3 bytes of payload
            0x02 -> send 50Hz filtered signals
            0x01 -> send IMU data streams
            0x01 -> send classifier events
        '''
        self.write_attr(0x19, b'\x01\x03\x02\x01\x01')

        '''Sending this sequence for v1.0 firmware seems to enable both raw data and
        pose notifications.
        '''

        '''By writting a 0x0100 command to handle 0x28, some kind of "hidden" EMG
        notification characteristic is activated. This characteristic is not
        listed on the Myo services of the offical BLE specification from Thalmic
        Labs. Also, in the second line where we tell the Myo to enable EMG and
        IMU data streams and classifier events, the 0x01 command wich corresponds
        to the EMG mode is not listed on the myohw_emg_mode_t struct of the Myo
        BLE specification.
        These two lines, besides enabling the IMU and the classifier, enable the
        transmission of a stream of low-pass filtered EMG signals from the eight
        sensor pods of the Myo armband (the "hidden" mode I mentioned above).
        Instead of getting the raw EMG signals, we get rectified and smoothed
        signals, a measure of the amplitude of the EMG (which is useful to have
        a measure of muscle strength, but are not as useful as a truly raw signal).
        '''

        # self.write_attr(0x28, b'\x01\x00')  # Not needed for raw signals
        # self.write_attr(0x19, b'\x01\x03\x01\x01\x01')

    def mc_start_collection(self):
        '''Myo Connect sends this sequence (or a reordering) when starting data
        collection for v1.0 firmware; this enables raw data but disables arm and
        pose notifications.
        '''

        self.write_attr(0x28, b'\x01\x00')  # Suscribe to EMG notifications
        self.write_attr(0x1d, b'\x01\x00')  # Suscribe to IMU notifications
        self.write_attr(0x24, b'\x02\x00')  # Suscribe to classifier indications
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')  # Set EMG and IMU, payload size = 3, EMG on, IMU on, classifier on
        self.write_attr(0x28, b'\x01\x00')  # Suscribe to EMG notifications
        self.write_attr(0x1d, b'\x01\x00')  # Suscribe to IMU notifications
        self.write_attr(0x19, b'\x09\x01\x01\x00\x00')  # Set sleep mode, payload size = 1, never go to sleep, don't know, don't know
        self.write_attr(0x1d, b'\x01\x00')  # Suscribe to IMU notifications
        self.write_attr(0x19, b'\x01\x03\x00\x01\x00')  # Set EMG and IMU, payload size = 3, EMG off, IMU on, classifier off
        self.write_attr(0x28, b'\x01\x00')  # Suscribe to EMG notifications
        self.write_attr(0x1d, b'\x01\x00')  # Suscribe to IMU notifications
        self.write_attr(0x19, b'\x01\x03\x01\x01\x00')  # Set EMG and IMU, payload size = 3, EMG on, IMU on, classifier off

    def mc_end_collection(self):
        '''Myo Connect sends this sequence (or a reordering) when ending data collection
        for v1.0 firmware; this reenables arm and pose notifications, but
        doesn't disable raw data.
        '''

        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')
        self.write_attr(0x19, b'\x09\x01\x00\x00\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x00\x01\x01')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')

    def vibrate(self, length):
        if length in xrange(1, 4):
            # first byte tells it to vibrate; purpose of second byte is unknown (payload size?)
            self.write_attr(0x19, pack('3B', 3, 1, length))

    def set_leds(self, logo, line):
        self.write_attr(0x19, pack('8B', 6, 6, *(logo + line)))

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)

    def add_imu_handler(self, h):
        self.imu_handlers.append(h)

    def add_pose_handler(self, h):
        self.pose_handlers.append(h)

    def add_arm_handler(self, h):
        self.arm_handlers.append(h)

    def add_battery_handler(self, h):
        self.battery_handlers.append(h)

    def createWindow(self,emg):
        self.currentSlide.append(emg)
        if(len(self.currentSlide) == 50):
            #print(self.currentSlide)
            temp=[]
            temp.append(self.currentSlide)
            self.currentSlide = self.currentSlide[25:]
            return temp[-1]
        else:
           pass

    def on_emg(self, emg, moving):
        global string
        # print('on_emg')
        for h in self.emg_handlers:
            h(emg, moving)
        emgvals= ""
        processed=""
        emgvals = self.createWindow(emg)
        #print(emgvals)
        if emgvals!=None:
                df = preprocess(emgvals)
                #print(df)
                np.set_printoptions(formatter={'float': '{:.5f}'.format})
                emg1_cA,emg1_cD = wavelet(df.emg1)
                emg2_cA,emg2_cD = wavelet(df.emg2)
                emg3_cA,emg3_cD = wavelet(df.emg3)
                emg4_cA,emg4_cD = wavelet(df.emg4)
                emg5_cA,emg5_cD = wavelet(df.emg5)
                emg6_cA,emg6_cD = wavelet(df.emg6)
                emg7_cA,emg7_cD = wavelet(df.emg7)
                emg8_cA,emg8_cD = wavelet(df.emg8)
                cA_total=itertools.chain(emg1_cA,emg2_cA,emg3_cA,emg4_cA,emg5_cA,emg6_cA,emg7_cA,emg8_cA)
                newlist_cA=list(cA_total)
                output=[]
                output=[round(float(i), 5) for i in newlist_cA]
                #print(np.shape(output))
                #cA_labelled = np.append([output],'lightest')
                #cA_labelled = np.append([cA_labelled],'1')

                #col = [str(x)+"c" for x in range(1,59)] # for classification
                col = [str(x)+"c" for x in range(1,57)]
                df2 = pd.DataFrame(columns = col)
                #df2.loc[len(df2)] = cA_labelled
                df2.loc[len(df2)] = output  # for prediction with no of col=57
                #classify(df2)
                #modelFit()

                r=prediction(df2)
                 # r = prediction(df2)

                myPredictions[r]+=1
                # print (reList)
                # print (lightList)
                # print (heaviestList)
                # l1=len(reList)
                # l2=len(lightList)
                # l3=len(heaviestList)

                # # print (l1,l2,l3)
                # if l1>l2 and l1>l3:
                #     print ("rest")
                # elif l2>l1 and l2>l3:
                #     print ("lightest")
                # else:
                #     print ("heaviest")

    def on_imu(self, quat, acc, gyro):
        for h in self.imu_handlers:
            h(quat, acc, gyro)

    def on_pose(self, p):
        for h in self.pose_handlers:
            h(p)

    def on_arm(self, arm, xdir):
        for h in self.arm_handlers:
            h(arm, xdir)

    def on_battery(self, battery_level):
        for h in self.battery_handlers:
            h(battery_level)


if __name__ == '__main__':

    try:
        import pygame
        from pygame.locals import *
        HAVE_PYGAME = True
    except ImportError:
        HAVE_PYGAME = False

    if HAVE_PYGAME:
        w, h = 800, 600
        scr = pygame.display.set_mode((w, h))

    last_vals = None

    def plot(scr, vals):
        DRAW_LINES = True

        global last_vals
        if last_vals is None:
            last_vals = vals
            return

        D = 5
        scr.scroll(-D)
        scr.fill((0, 0, 0), (w - D, 0, w, h))
        for i, (u, v) in enumerate(zip(last_vals, vals)):
            if DRAW_LINES:
                pygame.draw.line(scr, (0, 255, 0),
                                 (w - D, int(h/9 * (i+1 - u))),
                                 (w, int(h/9 * (i+1 - v))))
                pygame.draw.line(scr, (255, 255, 255),
                                 (w - D, int(h/9 * (i+1))),
                                 (w, int(h/9 * (i+1))))
            else:
                c = int(255 * max(0, min(1, v)))
                scr.fill((c, c, c), (w - D, i * h / 8, D, (i + 1) * h / 8 - i * h / 8))

        pygame.display.flip()
        last_vals = vals

    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)

    def proc_emg(emg, moving, times=[]):
        if HAVE_PYGAME:

            	plot(scr, [e / 500. for e in emg])
		#print('emg: ')
		#print([e / 500. for e in emg])
        else:
            pass
            # print(emg)

        # print framerate of received data
        times.append(time.time())
        if len(times) > 20:
            # print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0)

    def proc_battery(battery_level):
        # print("Battery level: %d" % battery_level)
        if battery_level < 5:
            m.set_leds([255, 0, 0], [255, 0, 0])
        else:
            m.set_leds([128, 128, 255], [128, 128, 255])
    # m.add_emg_handler(proc_emg)

    m.connect()

    m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
    # m.add_pose_handler(lambda p: print('pose', p))
    # m.add_imu_handler(lambda quat, acc, gyro: print('quaternion', quat))
    m.set_leds([128, 128, 255], [128, 128, 255])  # purple logo and bar LEDs
    m.vibrate(1)
    # a = [str(x)+"c" for x in range(0,59)]
    # a = ','.join(a)
    # fout=open("s_data.csv","a")
    # fout.write(a+"\n")
    # fout.close()
    startTime=datetime.datetime.now()

    try:
        while datetime.datetime.now().second - startTime.second  <= 1:
            m.run()
            if HAVE_PYGAME:
                for ev in pygame.event.get():
                    if ev.type == QUIT or (ev.type == KEYDOWN and ev.unicode == 'q'):

                        raise KeyboardInterrupt()

                    elif ev.type == KEYDOWN:
                        if K_1 <= ev.key <= K_3:
                            m.vibrate(ev.key - K_0)
                        if K_KP1 <= ev.key <= K_KP3:
                            m.vibrate(ev.key - K_KP0)

    except KeyboardInterrupt:
        pass
    finally:
        # m.disconnect()
        # print("Disconnected")
        # print (myPredictions)
        flag=False

        for i in addressList:
            if i==35:# address produced only when band is worn
                flag = True
                break
            else:
                pass
        if flag is True:  # band is worn
            print(max(myPredictions,key=myPredictions.get))
        else:
            print ('Not worn')
