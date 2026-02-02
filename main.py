import cv2
import numpy as np

class LineDetector:
    def __init__(self):
        pass

    def reseize(self, image):
        return_img_resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        return return_img_resized

    def region_of_interest(self, frame):
        x, y, w, h = (0, 180, 640, 300)
        if w > 0 and h > 0:
            # Crop image
            cropped_image = frame[y:y + h, x:x + w].copy()

        return cropped_image, y

    # def threshold_img(self, image):
    #     #preprocess
    #     bgr_img  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert image into bgr
    #     blur_img = cv2.GaussianBlur(bgr_img, (7,7), 0)
    #
    #     #thresh
    #     _, thresh = cv2.threshold(blur_img, 127, 255, cv2.THRESH_BINARY)
    #
    #     return thresh

    def normalization_gray(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16,16))

        norm = clahe.apply(gray)


        return norm

    # def counters_preprocess(self, thresh_img):
    #     hsv_image = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2HSV)
    #
    #     # bright colors detection
    #     lower_yellow = np.array([25, 100, 100])
    #     upper_yellow = np.array([35, 255, 255])
    #
    #     # black and white mask
    #     black_and_white_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)  # make for any bright color
    #
    #     return black_and_white_mask

    def detect_counters(self, black_and_white_mask):
        #finding contours
        contours, hierarchy = cv2.findContours(black_and_white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    # def draw_counters(self, image, counters):
    #     for counter in counters:
    #         if cv2.arcLength(counter, True) > 100:
    #             cv2.drawContours(image, counter, -1, (0, 255, 0), 2)
    #     return image

    def auto_canny(self, gray, sigma=0.33):
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(gray, lower, upper)

    def line_dection(self, black_and_white_mask):
        lines = cv2.HoughLinesP(black_and_white_mask, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)
        return lines

    def binary_from_sobel_x(self, gray, ksize=3, thresh=(40, 255)):
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        agx = np.absolute(gx)
        agx = (255 * agx / (agx.max() + 1e-6)).astype(np.uint8)
        lo, hi = thresh
        return cv2.inRange(agx, lo, hi)

    def filter_slanted_lines(self, lines, min_angle_deg=25, max_angle_deg=75):
        """
        Оставляет только наклонные линии (не горизонтальные).
        angle = atan2(dy, dx) в градусах, берем abs.
        """
        if lines is None:
            return None

        filtered = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue

            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if angle > 90:
                angle = 180 - angle

            if min_angle_deg <= angle <= max_angle_deg:
                filtered.append(l)

        return np.array(filtered) if len(filtered) else None

def apply_outside_lane_roi(mask):
    h, w = mask.shape[:2]
    poly = np.array([[
        (int(0.05*w), h),
        (int(0.40*w), int(0.45*h)),
        (int(0.60*w), int(0.45*h)),
        (int(0.95*w), h),
    ]], dtype=np.int32)

    roi = np.ones_like(mask) * 255
    cv2.fillPoly(roi, poly, 0)
    return cv2.bitwise_and(mask, roi)

def pick_two_best(lines, w, min_angle=25, max_angle=85):
    if lines is None:
        return None, None

    bestL, scoreL = None, -1
    bestR, scoreR = None, -1

    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        if not (min_angle <= angle <= max_angle):
            continue

        slope = dy / dx
        length = float(np.hypot(dx, dy))
        x_mid = 0.5 * (x1 + x2)

        score = length

        if slope < 0 and x_mid < w/2:
            if score > scoreL:
                bestL, scoreL = l, score

        if slope > 0 and x_mid > w/2:
            if score > scoreR:
                bestR, scoreR = l, score

    return bestL, bestR

def extend_simple(line, length=2000):
    x1, y1, x2, y2 = line[0]

    dx = x2 - x1
    dy = y2 - y1
    norm = np.sqrt(dx*dx + dy*dy)
    if norm == 0:
        return None

    dx /= norm
    dy /= norm

    x_start = int(x1 - dx * length)
    y_start = int(y1 - dy * length)
    x_end   = int(x2 + dx * length)
    y_end   = int(y2 + dy * length)

    return (x_start, y_start, x_end, y_end)

def main():
    line_detect   = LineDetector()
    image         = cv2.imread("/Users/arsenyzharkoy/PycharmProjects/MrStern/img.png")
    resized       = line_detect.reseize(image)
    resized_croped, y = line_detect.region_of_interest(resized)
    resized_clahe = line_detect.normalization_gray(resized_croped)

    #blur
    resized_clahe = cv2.GaussianBlur(resized_clahe, (7, 7), 0)
    #binary

    binary = line_detect.binary_from_sobel_x(resized_clahe)
    binary = apply_outside_lane_roi(binary)

    edges = line_detect.auto_canny(resized_clahe)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    lines = line_detect.line_dection(binary)
    lines = line_detect.filter_slanted_lines(lines)

    h, w = closed_edges.shape[:2]

    bestL, bestR = pick_two_best(lines, w)

    midpoints = []
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    for l in (bestL, bestR):
        if l is None:
            continue
        l = extend_simple(l)
        cv2.line(resized, (l[0], l[1]+y), (l[2], l[3]+y), (0, 255, 255), 3)

        x1, y1, x2, y2 = l

        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        mid_y += y

        mid_x = int(np.clip(mid_x, 0, resized.shape[1] - 1))
        mid_y = int(np.clip(mid_y, 0, resized.shape[0] - 1))

        midpoints.append((mid_x, mid_y))

    for center in midpoints:
        mid_x, mid_y = center
        cv2.circle(resized, (mid_x, mid_y), 8, (255, 0, 255), -1)
    try:
        point1, point2 = midpoints
        mid_x = (point1[0] + point2[0]) // 2
        mid_y = (point1[1] + point2[1]) // 2

        x, y = image.shape[:2]
        cv2.circle(resized, (mid_x, mid_y), 8, (255, 0, 255), -1)
        cv2.line(resized, (mid_x, 0), (mid_x, y), (255, 0, 255), 3)
    except:
        pass

    cv2.imshow("Original_img", binary)
    cv2.imshow("resized_img", resized)
    cv2.imshow("edges", edges)
    cv2.imshow("closed_edges", closed_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass

main()