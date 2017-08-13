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
        

draw = Drawing(img_color)
#draw.line((0,0), (150,150), (255,255,255), 15)
#draw.rectangle((15,25), (250,150), (0,255,0), 5)
#draw.circle((100,63), 55, (0,0,255), 5, 5)
#draw.circle((100,63), 55, (0,155,255), 5, 500)
points = [[10,5], [80,99], [70,200], [60, 50], [250,89]]
writing = 'this is writing'
font = cv2.FONT_HERSHEY_SIMPLEX
#draw.write(writing, (0,130), font, 1, (255,200,255), 2)

#draw.polygon(points, (255, 0, 0))

"""
points = [[10,5], [80,99], [70,200], [60, 50], [250,89]]
print np.array([[10,5], [80,99], [70,200], [60, 50], [250,89]], np.int32)
print points
)
y = np.zeros((len(points),2), dtype=np.int)
print y
i=0
while i < len(points):
    y[i] = points[i]
    i+=1
    
print y
print np.array(points)
y[1] = [1,1]
print y[1]
"""
