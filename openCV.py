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
        def color_filtering(self, lower, upper):
        """
        Allows you to filter an image for a selected color. lower argument is
        a list of three numbers representing lower range limit for
        Hue, Saturation, Value. e.g. [50, 0, 0]. upper argument has the same
        form and represents the upper range limit.
        """
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower = np.array(lower)
        upper = np.array(upper)
        
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(self.img, self.img, mask=mask)

        cv2.imshow('img', self.img)
        cv2.imshow('mask', mask)
        cv2.imshow('result', res)
        return res

    def blur(self, lower, upper, dim_one, dim_two, Gauss=False, Median=False):
        """
        lower argument is a list of three numbers representing lower range limit
        for Hue, Saturation, Value. e.g. [50, 0, 0]. upper argument has the same
        form and represents the upper range limit. dim_one and dim_ two are
        dimensions e.g. 15 and 15
        """
        res = self.color_filtering(lower, upper)
        kernel = np.ones((dim_one,dim_two), np.float32)/(dim_one*dim_two)
        smoothed = cv2.filter2D(res, -1, kernel)
        gaussian = cv2.GaussianBlur(res, (dim_one, dim_two), 0)
        med = cv2.medianBlur(res, dim_one)

        if Gauss == True and Median == True:
            cv2.imshow('median', med)
            cv2.imshow('gaussian', gaussian)
        elif Gauss == True and Median == False:            
            cv2.imshow('gaussian', gaussian)
        elif Median == True and Gauss == False:
            cv2.imshow('median', med)
        else:
            cv2.imshow(smoothed)

    def erosion_or_dilation(self, lower, upper, window):
        """
        erosion has a slider you decide the size of. If all pixels are identical
        in color, then it moves on. If not, it'll remove the rogue pixel.
        dilation pushes out until it can't go any further. lower argument is a
        list of three numbers representing lower range limitfor Hue, Saturation,
        Value. e.g. [50, 0, 0]. upper argument has the same form and represents
        the upper range limit. kernel is a tuple of dimensions for the miniature
        window. e.g. (5,5)
        """
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower = np.array(lower)
        upper = np.array(upper)

        kernel = np.ones(window, np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(self.img, self.img, mask=mask)
        unmasked = cv2.bitwise_and(self.img,self.img)
        
        unmasked_ero = cv2.erode(unmasked, kernel, iterations=1)
        unmasked_dil = cv2.dilate(unmasked, kernel, iterations=1)
        erosion = cv2.erode(mask, kernel, iterations=1)
        dilation = cv2.dilate(mask, kernel, iterations=1)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('erosion', erosion)
        cv2.imshow('umasked_erosion', unmasked_ero)
        cv2.imshow('dilation', dilation)
        cv2.imshow('unmasked_dilation', unmasked_dil)

        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)

        return kernel, res, unmasked, mask


