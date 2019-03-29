#!/usr/bin/python3
import numpy as np
import cv2
import json
import argparse

# Example ussage.
# python3 config_file cam_number distance_to_plane
# python3 calibration.json 1 200.0

# Configure parser. Define arguments for config file, camera number and distance.
parser = argparse.ArgumentParser(description='2D distance calculator.')
parser.add_argument('config_file', metavar='config', type=str, nargs=1, help='Configuration file (json) containing intrinsic and extrinsic properties of the camera.')
parser.add_argument('camera_number', metavar='cam', type=int, nargs=1, help='Camera number for OpenCV.')
parser.add_argument('distance', metavar='distance', type=float, nargs=1, help='Distance (cm) from the camera to the plane to be measured.')

# Read arguments. Whenever a parameter is not provided execution stops, caused by the function
# parser.parse_args.
args = parser.parse_args()
CONFIG_FILE = args.config_file[0]
DISTANCE_TO_PLANE = args.distance[0]
CAM_NUMBER = args.camera_number[0]

# Read configuration file.
data = {}
with open(CONFIG_FILE, 'r') as fh:
    data = json.load(fh)

# Constant definitions.
FOCAL_DISTANCE = data['focal_length']   # Reads the focal distance from the config file.
FRAME_WIDTH = data['width']             # Reads the width from the config file.
FRAME_HEIGHT = data['height']           # Reads the height from the config file.
CX = data['cx']                         # Reads the -x center of frame from the config file.
CY = data['cy']                         # Reads the -y center of frame from the config file.
MARKER_COLOR = (0, 255, 125)            # Defines the colors of the markers. Default is green.
CROSSHAIR_COLOR = (255, 0, 255)         # Defines the colors of the corsshair. Default is magenta.
FONT_COLOR = (255, 255, 0)              # Defines the color of the font. Default is cyan.
FONT = cv2.FONT_HERSHEY_SIMPLEX         # Font style for OpenCV.
FONT_THICKNESS = 1                      # Font thickness parameter for OpenCV.
FONT_SCALE = 0.6                        # Font scale parameter for OpenCV.
MARKER_STYLE = 2                        # Defines the marker style. 2 = corsshair, 1 or other = box.

class DimentionCalculator:

    def __init__(self):
        self.mouse = (0, 0)
        self.points = []

    def event(self, event, x, y, flags, param):
        """
            Callback function for cv2.setMouseCallback.
        """

        # Left button released.
        if event == cv2.EVENT_LBUTTONUP:

            # Make sure only two points can be stored.
            if (len(self.points) > 1): self.points = []

            # Store last clicked point.
            self.points.append((x, y))

        # Mouse moved.
        elif event == cv2.EVENT_MOUSEMOVE:

            # Store last known position.
            self.mouse = (x, y)

    def getZoomedSection(self, img, r=50, scale=5):
        """
            Returns a zoomed and cropped section around the mouse's position.
            The box has a side length of 2 * r.
        """    

        # Calculate the center x position of the box.
        # Constrains it to be within the image limits.
        x = self.mouse[0]
        if (x - r < 0):
            x = r
        elif (x + r > FRAME_WIDTH):
            x = FRAME_WIDTH - r

        # Calculate the center y position of the box.
        # Constrains it to be within the image limits.
        y = self.mouse[1]
        if (y - r < 0):
            y = r
        elif (y + r > FRAME_HEIGHT):
            y = FRAME_HEIGHT - r

        # Obtains the cropped image centered at (x, y) with a side length of 2 * r.
        cropped = img[y-r:y+r, x-r:x+r]

        # Return scaled image.
        return cv2.resize(cropped, (0, 0), fx=scale, fy=scale) 

    def midpoint(self, p1, p2):
        """
            Returns a tuple containing the midpoint between p1 and p2.
        """
        
        return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)

    def drawPoints(self, img):
        """
            Draw a crosshair for every point. If two points are stored, a line can be
            drawn.
        """

        # Draw crosshairs for every point.
        for p in self.points:
            self.drawCorsshair(img, p, color=MARKER_COLOR)

        # If two points are available.
        if (len(self.points) > 1):

            # Connect points with a line.
            cv2.line(img, self.points[0], self.points[1], MARKER_COLOR, 1)

            # Calculate real world and pixel distance.
            p1 = np.array(self.points[0])
            p2 = np.array(self.points[1])
            dist_world = np.linalg.norm((p1 - p2) * DISTANCE_TO_PLANE / FOCAL_DISTANCE)
            dist_px = np.linalg.norm(p1 - p2)

            # Draw calculated distance in cm.
            (mX, mY) = self.midpoint(self.points[0], self.points[1])
            self.drawText(img, "%.1fpx" % (dist_px), int(mX), int(mY) + 30)
            self.drawText(img, "%.3fcm" % (dist_world), int(mX), int(mY))

    def drawCorsshair(self, img, pos, r=10, color=CROSSHAIR_COLOR):
        """
            Draws a crosshair centered at pos (x, y) with a line length of 2 * r.
        """
        
        # Draws a 3px box around the desired position.
        if (MARKER_STYLE == 1):

            r = 3
            box_coords = (
                (pos[0] - r, pos[1] - r),
                (pos[0] + r, pos[1] + r)
            )
            cv2.rectangle(img, box_coords[0], box_coords[1], color)
        
        # Draws a crosshair centered at the desired position.
        else :

            # Draw vertical line of the crosshair.
            p1 = (pos[0], pos[1] - r)
            p2 = (pos[0], pos[1] + r)
            cv2.line(img, p1, p2, color, 1)

            # Draw horizontal line of the crosshair.
            p1 = (pos[0] - r, pos[1])
            p2 = (pos[0] + r, pos[1])
            cv2.line(img, p1, p2, color, 1)

    def drawText(self, img, text, x0, y0):
        """
            Draws text with a black background at top left coordinate x0, y0.
        """ 

        # Calculate text dimentions.
        (width, height) = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]

        # Calculate box dimentions.
        box_coords = (
            (x0, y0 - 5), 
            (x0 + width, y0 + height + 5)
        )

        # Draw background box.
        cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)

        # Position is set by the bottom left corner.
        cv2.putText(img, text, (x0, y0 + height), FONT, fontScale=FONT_SCALE, color=FONT_COLOR, thickness=FONT_THICKNESS)

    def drawStatusInfo(self, img):
        """
            Draws status information on the image.
        """

        # Formated text for the mouse coordinates.
        mouse_coordinates = '(%d, %d)' % self.mouse
        
        # Formated text for the first point. Blank if not set.
        p1_coordinates = 'P1: -'
        if (len(self.points) > 0):
            p1_coordinates = 'P1: (%d, %d)' % self.points[0]

        # Formated text for the second point. Blank if not set.
        p2_coordinates = 'P2: -'
        if (len(self.points) > 1):
            p2_coordinates = 'P2: (%d, %d)' % self.points[1]

        # Draw mouse coordinates.
        self.drawText(img, mouse_coordinates, 10, 15)
        self.drawText(img, p1_coordinates, 10, 45)
        self.drawText(img, p2_coordinates, 10, 70)
        self.drawText(img, "Distance: %.1fcm" % (DISTANCE_TO_PLANE), 10, 100)
        self.drawText(img, "f: %.2f" % (FOCAL_DISTANCE), 10, 130)

def main():
    """
        Main function.
    """

    # Initialize calculator.
    calc = DimentionCalculator()

    # Define cv2 windows and set their position.
    cv2.namedWindow('input', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('input', 50, 200)
    cv2.namedWindow('zoomed', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('zoomed', 50 + FRAME_WIDTH, 200)

    # Set callback for mouse events.
    cv2.setMouseCallback('input', calc.event)

    # Open webcam and set frame properties.
    cap = cv2.VideoCapture(CAM_NUMBER)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Main loop.
    while(True):

        # Read input image.
        ret, img = cap.read()

        # Make sure camera is connected and retrieving data.
        if (ret):

            # Main pipeline.
            calc.drawCorsshair(img, calc.mouse)
            calc.drawPoints(img)
            zoomed = calc.getZoomedSection(img)
            calc.drawStatusInfo(img)

            # Show images.
            cv2.imshow('input', img)  
            cv2.imshow('zoomed', zoomed)  

            # Check if q is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:

            break

    # Clear windows and webcam.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()