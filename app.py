from scipy.spatial import distance
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import base64
import numpy as np
import cv2

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

colors = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_closest_circle_in_group(circle, group):
    distances = [(euclidean_distance((circle[0], circle[1]), (c[0], c[1])), i) for i, c in enumerate(group)]
    closest_circle = min(distances, key=lambda x: x[0])
    return closest_circle[1]

def identify_group_and_closest_circle(circle, groups):
    group_distances = [(euclidean_distance((circle[0], circle[1]), (c[0], c[1])), i, j)
                       for i, group in enumerate(groups) for j, c in enumerate(group)]
    closest_group_circle = min(group_distances, key=lambda x: x[0])
    closest_group_index = closest_group_circle[1]
    closest_circle_index = find_closest_circle_in_group(circle, groups[closest_group_index])
    return closest_group_index, closest_circle_index

def group_circles_by_lines(circles, lines):
    grouped_circles = []
    current_group = []
    line_index = 0

    for circle in circles:
        circle_y = circle[1]

        if line_index == 0 and circle_y < lines[line_index][1]:
            current_group.append(circle)
        else:
            while line_index < len(lines) and circle_y >= lines[line_index][1]:
                grouped_circles.append(current_group)
                current_group = []
                line_index += 1
            current_group.append(circle)

    grouped_circles.append(current_group)
    return grouped_circles

def detect_all_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 5)
    ret, thresh2 = cv2.threshold(blurred, 140, 140, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blurred, 5, 40)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=5,
                               param1=20, param2=7, minRadius=2, maxRadius=5)

    circle_coords = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            circle_coords.append((x, y, r))

    return image, circle_coords

def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 2)
    ret, thresh2 = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    circles = cv2.HoughCircles(thresh2, cv2.HOUGH_GRADIENT, dp=1.2, minDist=blurred.shape[0] / 8,
                               param1=50, param2=9, minRadius=2, maxRadius=7)

    circle_coords = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            circle_coords.append((x, y, r))

    return image, circle_coords

def detect_horizontal_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_coords = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 200 < cv2.contourArea(contour) < 2000 and w > 50 and h < 20:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            line_coords.append((x, y, w, h))

    return image, line_coords

def get_roi(origin_binary):
    contours, _ = cv2.findContours(origin_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_squares = []

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if 7 < len(approx) < 10:
            area = cv2.contourArea(contour)
            if 40 < area < 80:
                black_squares.append(approx)

    centers = []
    for square in black_squares:
        M = cv2.moments(square)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

    if not centers:
        return {'x_min': -1, 'y_min': -1, 'x_max': -1, 'y_max': -1}
    centers = np.array(centers)
    x_min = int(np.min(centers[:, 0]))
    y_min = int(np.min(centers[:, 1]))
    x_max = int(np.max(centers[:, 0]))
    y_max = int(np.max(centers[:, 1]))
    return {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_data = data['image']
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        origin_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        origin_binary = cv2.adaptiveThreshold(origin_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        roi_result = get_roi(origin_binary)

        processed_frame = frame[roi_result['y_min']:roi_result['y_max'], roi_result['x_min']:roi_result['x_max']]

        if roi_result['x_min'] == -1:
            return jsonify({"error": "No ROI found"}), 400

        line_image, line_coords = detect_horizontal_lines(processed_frame.copy())
        circle_image, circle_coords = detect_circles(processed_frame.copy())

        circles_line = processed_frame[:, int(processed_frame.shape[1]*0.1): int(processed_frame.shape[1]*0.15)]
        circle_all_image, circle_all_coords = detect_all_circles(circles_line)

        circle_coords.sort(key=lambda x: x[1])
        circle_all_coords.sort(key=lambda x: x[1])
        line_coords.sort(key=lambda x: x[1])
        grouped_circles = group_circles_by_lines(circle_all_coords, line_coords)

        results = []
        for circle in circle_coords:
            group_index, circle_index = identify_group_and_closest_circle(circle, grouped_circles)
            results.append((group_index, circle_index))

        for i, group in enumerate(grouped_circles):
            color = colors[i % len(colors)]
            for circle in group:
                x, y, r = circle
                cv2.circle(processed_frame, (int(x)+int(processed_frame.shape[1]*0.1), int(y)), int(r), color, 5)

        _, buffer = cv2.imencode('.png', processed_frame)
        processed_img_str = base64.b64encode(buffer).decode('utf-8')

        answers = [2, 3, 1, 3]
        final_result = 0
        for i, (group_index, circle_index) in enumerate(results):
            if circle_index + 1 == answers[i]:
                final_result += 1

        return jsonify({
            "image": f"data:image/jpeg;base64,{processed_img_str}",
            "result": {
                "final_result": final_result,
                "total": len(grouped_circles)
            }
        })
    except Exception as e:
        print('This error is: ', e)

if __name__ == '__main__':
    app.run(debug=True)
