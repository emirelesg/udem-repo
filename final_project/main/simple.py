#!/usr/bin/python3
from imutils.video.pivideostream import PiVideoStream
import cv2
import numpy as np
import time
from processor import ImageProcessor
from car import Car
from tracker import Tracker
from pid import PID


pidLeft = PID(kp=0.5, ki=0, kd=0, setPoint=0)
pidRight = PID(kp=-0.5, ki=0, kd=0, setPoint=0)
pid = PID(kp=0.25, ki=0, kd=0, setPoint=0)

averageLeft = np.poly1d(np.array([-3.45, 778.36]))
averageRight = np.poly1d(np.array([3.66, -328.14]))

car = Car('/dev/ttyAMA0', 115200)
car.start()
processor = ImageProcessor((480, 320), 20)
cv2.namedWindow('main', cv2.WINDOW_AUTOSIZE)
camera = PiVideoStream(resolution=processor.frameDimensions, framerate=processor.frameRate).start()
time.sleep(2)
command = ''
rightTracker = Tracker()
leftTracker = Tracker()

def writeCarStatus(frame, status):
    if status:
        cv2.putText(frame, 'RPi: %.2f v' % (status['rpiBatteryVoltage']), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, 'Motor: %.2f v' % (status['motorBatteryVoltage']), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    return frame

def main():

    global command, camera, processor
    statusRequestTime = time.time()
    latestCarStatus = None
    enableControl = False
    enableMotors = False

    pid = 0
    sumError = 0
    averageError = 0
    prevError = 0
    dt = 1.0 / 20

    while True:

        frame = camera.read()
        out = processor.process(frame)
        
        left = leftTracker.add(processor.left.poly)
        right = rightTracker.add(processor.right.poly)

        # processor.drawPoly(out, processor.left.poly, (0, 100, 200))
        # processor.drawPoly(out, processor.right.poly, (200, 100, 0))

        processor.drawPoly(out, left, (0, 50, 255))
        processor.drawPoly(out, right, (255, 50, 0))
        
        processor.drawPoly(out, averageLeft, (255, 255, 255))
        processor.drawPoly(out, averageRight, (255, 255, 255))

        if enableControl:
            y0 = processor.roiY[0] * processor.h
            leftError = averageLeft(y0) - left(y0)
            rightError = averageRight(y0) - right(y0)
            cv2.putText(out, '%.2f' % (leftError), (int(averageLeft(y0)), int(y0) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(out, '%.2f' % (rightError), (int(averageRight(y0)), int(y0) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            kp = 0.05
            ki = 0.05
            kd = 0.01
            prevError = averageError
            averageError = (leftError + rightError) / 2
            
            sumError += averageError * ki * dt
            if sumError > 50:
                sumError = 50
            elif sumError < -50:
                sumError = -50

            p = kp * averageError
            i = sumError
            d = (averageError - prevError) * kd / dt

            pid = p + i + d


            baseSpeed = 50
            velLeft = int(baseSpeed - pid)
            velRight = int(baseSpeed + pid)

            if velLeft < 0:
                velLeft = 0
            elif velLeft > 100:
                velLeft = 100
            if velRight < 0:
                velRight = 0
            elif velRight > 100:
                velRight = 100

            if enableMotors:
                command = 'VEL %d %d \t PID %.1f P %.1f I %.1f D %.1f' % (velLeft, velRight, pid, p, i ,d)
                print(command)


        out2 = writeCarStatus(out, latestCarStatus)
        cv2.imshow('main', out2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('k'):
            print('Left: ', processor.left.poly.coeffs)
            print('Right: ', processor.right.poly.coeffs)
        elif key == ord('w'):
            command = 'VEL 50 50'
        elif key == ord('s'):
            command = 'VEL -50 -50'
        elif key == ord('a'):
            command = 'VEL 0 50'
        elif key == ord('d'):
            command = 'VEL 50 0'
        elif key == ord(' '):
            command = 'VEL 0 0'
        elif key == ord('c'):
            enableControl = not enableControl
            if enableControl:
                print('CONTROL ON')
            else:
                print('CONTROL OFF')
            enableMotors = False
        elif key == ord('m'):
            enableMotors = not enableMotors
            if enableMotors:
                print('MOTORS ON')
            else:
                print('MOTORS OFF')
                command = 'VEL 0 0'


        car.commandQ.put(command)

        if time.time() - statusRequestTime > 1.0:
            car.requestStatus.set()
            statusRequestTime = time.time()
        
        while not car.messageQ.empty():
            latestCarStatus = car.messageQ.get()

    car.stop.set()
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()