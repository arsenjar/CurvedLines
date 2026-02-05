import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

# install opencv-contrib

class Curved_lines_detector:
    def __init__(self):
        pass

    # getting binary mask for the lines
    def threshold_img(self, image):
        # getting image as an input
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 10, 150])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return mask #binary mask

    def frame_preprocessing(self, frame):
        # resizing the image
        resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        #blur
        resized_blured = cv2.blur(resized, (9, 9))
        resized_blured = cv2.GaussianBlur(resized_blured, (11, 11), 0)
        black_and_white_mask = self.threshold_img(resized_blured)

        # morphology
        kernel = np.ones((5, 5), np.uint8)
        black_and_white_mask = cv2.morphologyEx(black_and_white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        black_and_white_mask = cv2.morphologyEx(black_and_white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        #detection
        skeleton = cv2.ximgproc.thinning(black_and_white_mask)

        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats((skeleton > 0).astype(np.uint8), connectivity=8)
        # skeleton > 0 - True line pixel
        # num2 - number of connected components
        # labels2 - same size matrix
        # stats2 - component statistics
        # Getting 2 big components
        areas = stats2[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1]  #sorting down
        top = order[:2] + 1  # we exclude background

        line_clusters = []
        for lab in top:
            ys, xs = np.where(labels2 == lab)
            pts = np.array(list(zip(xs, ys)), dtype=np.int32) # optimized array
            line_clusters.append(pts)

        colors = [(255, 0, 0), (0, 255, 0)]  # BGR
        for i, cluster in enumerate(line_clusters):
            for x, y in cluster:
                cv2.circle(resized, (int(x), int(y)), 2, colors[i], -1)

        # mean 1 line
        mean_x0 = float(np.mean(line_clusters[0][:, 0]))
        mean_y0 = float(np.mean(line_clusters[0][:, 1]))
        # mean 2 line
        mean_x1 = float(np.mean(line_clusters[1][:, 0]))
        mean_y1 = float(np.mean(line_clusters[1][:, 1]))

        #middle point
        mid_x = (mean_x0 + mean_x1) / 2.0
        mid_y = (mean_y0 + mean_y1) / 2.0

        cv2.circle(resized, (int(mid_x), int(mid_y)), 8, (255, 255, 255), -1) # dot

        l1 = line_clusters[0][np.argsort(line_clusters[0][:, 1])]
        l2 = line_clusters[1][np.argsort(line_clusters[1][:, 1])]

        map1 = {}
        map2 = {}
        #using dictionary in order to exclude values with the same y
        for x, y in l1:
            map1.setdefault(y, []).append(x)

        for x, y in l2:
            map2.setdefault(y, []).append(x)

        common_ys = sorted(set(map1.keys()) & set(map2.keys()))

        mid_points = []

        for y in common_ys[::4]:  # шаг 4 для разрежения
            x1 = int(np.mean(map1[y]))
            x2 = int(np.mean(map2[y]))

            mid_x = int((x1 + x2) / 2)
            mid_points.append((mid_x, y))

        # # smoothed line
        # window_size = 3
        # window = np.ones(window_size) / window_size
        # smoothed_data = np.convolve(mid_points, window, mode='valid')
        xs = np.array([p[0] for p in mid_points])
        ys = np.array([p[1] for p in mid_points])

        xs_s = uniform_filter1d(xs, size=7)
        ys_s = uniform_filter1d(ys, size=7)

        smoothed_data = list(zip(xs_s.astype(int), ys_s.astype(int)))
        # рисуем middle line
        for i in range(len(mid_points) - 1):
            cv2.line(resized, smoothed_data[i], smoothed_data[i + 1], (0, 0, 255), 2)


        return resized

def main():
    cap = cv2.VideoCapture("/Users/arsenyzharkoy/PycharmProjects/MrStern/1000010201(1).mp4")

    lines_dt = Curved_lines_detector()

    while True:
        ret, frame = cap.read()

        if not cap.isOpened():
            raise RuntimeError('Cannot read from camera')


        out = lines_dt.frame_preprocessing(frame)

        cv2.imshow("frame", out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
