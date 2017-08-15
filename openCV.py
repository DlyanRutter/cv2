import cv2, os
import numpy as np
import matplotlib.pyplot as plt

idee = 2
x = '/users/dylanrutter/Downloads/test'
img_gray = cv2.imread(os.path.join(x, str(idee) + '.jpg'), cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(os.path.join(x, str(idee) + '.jpg'), cv2.IMREAD_COLOR)
img_color = cv2.resize(img_color, (0,0), fx=0.3, fy=0.3)
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
ROI = img_color[100:150, 100:150]
cv2.imshow('image',ROI)


def cam_videos():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
    
        cv2.imshow('frame',frame)
        cv2.imshow('gray', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() #releases camera
    cv2.destroyAllWindows()

def resize(image, width, show=True):
    """
    Takes in an image and a desired pixel width. If show is set to True,
    resized image will display.
    """
    r = float(float(width) / float(image.shape[1]))
    dimensions = (int(width), int(image.shape[0] * r))
    resized = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    if show == True:
        cv2.imshow("resized", resized)
    return resized

def flip(image):
    """
    flips an image vertically
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
 
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

class Drawing(object):
    def __init__(self, img):
        self.img = img

    def line(self, start, end, color, width=None, title='image', destroy=True):
        """
        start = starting coordinates e.g. (0,0), end = end coordinates
        e.g. (150, 150), color = color e.g. (255, 255, 255), width = line width
        e.g. 15. Setting destroy = True will erase image after you push a key
        """
        line = cv2.line(self.img, start, end, color, width)
        if destroy == False:
            cv2.imshow(title, self.img)
        if destroy == True:
            cv2.imshow(title, self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows

    def rectangle(self, top_left, bot_right, color, width):
        """
        top_left = top_left coordinate e.g. (15,25), bot_right = bottom right
        coordinate e.g. (200,150), width = width, color = color e.g. (0,0,0)
        """
        rectangle = cv2.rectangle(self.img, top_left, bot_right, color, width)
        cv2.imshow('rectangle', self.img)

    def circle(self, center, radius, color, thickness=1, shift=0):
        """
        center = center coordinates e.g. (100,63), radius = radius e.g. 55,
        color = color e.g. (0,0,255), fill_in = how much to fill in, -1 fills
        in circle, anything positive increases thickness. Shift is the number
        of fractional bits in the coordinates of the center and in the radius
        value
        """
        circle = cv2.circle(self.img, center, radius, color, thickness, shift)
        cv2.imshow('circle', self.img)

    def polygon(self, points, color):
        """
        points is a list of points with form [x,y],  Flag indicating whether
        the drawn polylines are closed or not. If they are closed, the
        function draws a line from the last vertex of each curve to its first
        vertex. color = color e.g. (0,255,0)
        """
        i = 0
        matrix = np.zeros((len(x),2), dtype=np.int32)
        while i < len(points):
            matrix[i] = points[i]
            i+=1
        matrix.reshape((-1,1,2))
        polygon = cv2.polylines(self.img, [matrix], True, (255,0,0),\
                                thickness=3)
        cv2.imshow('polygon', self.img)           

    def write(self, text, start, font_name, size, color, thickness=2):
        """
        text is a string of text, start is starting coordinates e.g. (0, 130),
        font_name is name of font e.g. cv2.FONT_HERSHEY_SIMPLEX, size is font
        size e.g. 1, color is color e.g. (0,0,0), Thickness is spacing thickness
        between letters e.g. 2.
        """
        cv2.putText(self.img, text, start, font_name, size, color, thickness)
        cv2.imshow('show', self.img)

    
class Threshold(object):
    def __init__(self, img):
        self.img = img

    def low_light(self, img):
        """
        used for low light images. Any pixel below 12 will be made
        black. Maximum pixel value is 255
        """
        retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                     cv2.THRESH_BINARY, 115, 1)
        cv2.imshow('low_light', gaus)
