#!/usr/bin/python3

import cv2
import numpy as np
from time import sleep

TRACK_VIDEO = '../fig/videos/running_track.mp4'
HIGHWAY_VIDEO = '../fig/videos/highway.mp4'
SCALE = 0.35

settings = {
    'video_file': HIGHWAY_VIDEO,
    'scale': SCALE,
    'blur_kernel_size': (7, 7),
    'blur_iterations': 1,
    'canny_low_threshold': 10,
    'canny_high_threshold': 40,
    'hough_rho': 1,
    'hough_threshold': 20,
    'hough_theta': np.pi / 180,
    'hough_min_line_length': 5,
    'hough_max_line_gap': 60,
    'show_all_hough_lines': False,
    'show_canny': False,
    'roi_vertices': np.array([[
        (430, 830),           # Bottom left roi coordinate.
        (900, 600),           # Top left roi coordinate.
        (1020, 600),          # Top Right roi coordinate.
        (1530, 830)           # Bottom right roi coordinate.
    ]]) * 0.5 * SCALE,
    'abs_min_line_angle': 20,
    'lane_min_y': int(600 * 0.5 * SCALE),
    'lane_max_y': int(830 * 0.5 * SCALE)
}

def isGrayscale(img):
    """
        Returns true if the image is grayscale (channels == 1).
        Returns false if channels > 1.
    """
    
    # If img.shape has a channel value, read it and determine if
    # it is a grayscale image. If it doesn't have, assume that the
    # image is grayscale.
    if (len(img.shape) == 3):
        return img.shape[2] == 1
    return True

def createRoi(img, vertices):
    """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
    """
    
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if isGrayscale(img):
        ignore_mask_color = 255
    else:
        ignore_mask_color = (255, 255, 255)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    return cv2.bitwise_and(img, mask)

def findLanes(hough_lines, img, settings):
    """
        Given an array of hough lines, detects which lines correspond to the left and right lane.
    """

    # Empty arrays for storing line coordinates (sets of points (x, y)).
    right_lines = {
        'x': [],
        'y': []
    }
    left_lines = {
        'x': [],
        'y': []
    }
    
    # Stores the fitted line coordinates for both lanes.
    # Its format is [x0, y0, x1, y1].
    lanes = {
        'left': [],
        'right': []
    }
    
    # Make sure at least some lines have been detected.
    if (type(hough_lines) == type(np.array([]))):

        # Iterate through every line.
        for line in hough_lines:

            # Usualy every line object contains a single set of coordinates,
            # nonetheless, it is placed inside a for for safety.
            for x1, y1, x2, y2 in line:

                # Calculate the direction of the line found.
                direction = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))

                # Only draw lines whose angle is greater than the threshold.
                if (np.abs(direction) > settings['abs_min_line_angle']):

                    # If lines have a positive direction they are from the
                    # right lane.
                    if (direction > 0):
                        # Right lane.
                        right_lines['x'].extend([x1, x2])
                        right_lines['y'].extend([y1, y2])
                        if (settings['show_all_hough_lines']): 
                            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    else:
                        # Left lane.
                        left_lines['x'].extend([x1, x2])
                        left_lines['y'].extend([y1, y2])
                        if (settings['show_all_hough_lines']):
                            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    
                    # Write the angle of the line for debugging purposes.
                    if (settings['show_all_hough_lines']): cv2.putText(img, '%.1f' % (direction), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 0), thickness=1)
    
    # Make sure points on the left side were found.
    if (len(left_lines['x']) > 0 and len(left_lines['y']) > 0) and not settings['show_all_hough_lines']:
        
        # Using the points found, find a 1st order polynomial that best fits the data.
        poly_left = np.poly1d(np.polyfit(left_lines['y'], left_lines['x'], deg=1))
        
        # Evaluate the function found for the desired lane lengths.
        lanes['left'] = [
            int(poly_left(settings['lane_min_y'])),
            settings['lane_min_y'],
            int(poly_left(settings['lane_max_y'])),
            settings['lane_max_y']
        ]
    
    # Make sure points on the right side were found.
    if (len(right_lines['x']) > 0 and len(right_lines['y']) > 0) and not settings['show_all_hough_lines']:
        
        # Using the points found, find a 1st order polynomial that best fits the data.
        poly_right = np.poly1d(np.polyfit(right_lines['y'], right_lines['x'], deg=1))
        
        # Evaluate the function found for the desired lane lengths.
        lanes['right'] = [
            int(poly_right(settings['lane_min_y'])), 
            settings['lane_min_y'], 
            int(poly_right(settings['lane_max_y'])), 
            settings['lane_max_y']
        ]

    return lanes

def drawLanes(lanes, img):
    """
        Given the output of findLanes, draws the lanes in the image.
    """
    
    if (len(lanes['left'])):
        cv2.line(img, (lanes['left'][0], lanes['left'][1]), (lanes['left'][2], lanes['left'][3]), (0, 0, 255), 2)
    
    if (len(lanes['right'])):
        cv2.line(img, (lanes['right'][0], lanes['right'][1]), (lanes['right'][2], lanes['right'][3]), (0, 0, 255), 2)

def findLines(img, settings):
    """
        Given an BGR image, finds all lines in the image using a Probabilistic Hough Transform.
    """
    
    # Convert to gray scale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur image so that the Canny edge detector doesn't find useless edges.
    blur_gray = gray
    for i in range(settings['blur_iterations']):
        blur_gray = cv2.GaussianBlur(blur_gray, (settings['blur_kernel_size']), sigmaX=0, sigmaY=0)
    
    # Detect edges and create a ROI for the section defined.
    edges = cv2.Canny(blur_gray, settings['canny_low_threshold'], settings['canny_high_threshold'], apertureSize=3)
    masked_edges = createRoi(edges, settings['roi_vertices'].astype(int))
    
    # Detect lines.
    hough_lines = cv2.HoughLinesP(
        masked_edges, 
        rho = settings['hough_rho'], 
        theta = settings['hough_theta'], 
        threshold = settings['hough_threshold'], 
        lines = np.array([]), 
        minLineLength = settings['hough_min_line_length'], 
        maxLineGap = settings['hough_min_line_length']
    )
    
    # Return all data.
    return  {
        'gray': gray,
        'blur_gray': blur_gray,
        'edges': edges,
        'masked_edges': masked_edges,
        'hough_lines': hough_lines
    }

# Open selected video file.
cap = cv2.VideoCapture(settings['video_file'])

# Get scaled video properties.
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * settings['scale'])
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * settings['scale'])
FRAME_FPS = cap.get(cv2.CAP_PROP_FPS)
print('Dimensions: %dx%d' % (FRAME_WIDTH, FRAME_HEIGHT))

# Create input and ROI windows. Place each window side by side.
cv2.namedWindow('input', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('input', 0, 200)
if (settings['show_canny']):
	cv2.namedWindow('ROI', cv2.WINDOW_AUTOSIZE)
	cv2.moveWindow('ROI', FRAME_WIDTH, 200)

# Open video.
while(cap.isOpened()):
    
    # Read frame form video.
    t0 = cv2.getTickCount()
    ret, frame = cap.read()
    
    # If video ended exit.
    if ret:
		
        # Resize frame to desired scale.
        if (settings['scale'] != 1):
            img = cv2.resize(frame, (0, 0), fx=settings['scale'], fy=settings['scale'])
        else:
            img = frame
        
        # Find all lines in the image.
        e1 = cv2.getTickCount()
        output = findLines(img, settings)
        e2 = cv2.getTickCount()
        dt_findLines = (e2 - e1) / cv2.getTickFrequency()
        
        # With the lines found, find which of them belong to the left and right lanes.
        e1 = cv2.getTickCount()
        lanes = findLanes(output['hough_lines'], img, settings)
        e2 = cv2.getTickCount()
        dt_findLanes = (e2 - e1) / cv2.getTickFrequency()
        
        # Draw results.
        drawLanes(lanes, img)
        
        # Calculate loop time.
        e2 = cv2.getTickCount()
        dt = (e2 - t0) / cv2.getTickFrequency()
        
        # Draw times to output image.
        cv2.putText(img, 'Cycle: %.4fms - %.1ffps' % (dt, 1 / dt), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(img, 'findLines: %.4fms' % (dt_findLines), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(img, 'findLanes: %.4fms' % (dt_findLanes), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('input', img)
        if (settings['show_canny']):
			cv2.imshow('ROI', output['masked_edges'])

    else:
        break
        
    # If 'q' is pressed, then exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Clear memory and windows.
cap.release()
cv2.destroyAllWindows()
