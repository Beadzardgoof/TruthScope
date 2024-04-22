import argparse
from facenet_pytorch import MTCNN
import cv2
import mediapipe as mp
from ffpyplayer.player import MediaPlayer
from datetime import datetime
from matplotlib import pyplot as plt
import mss
import os
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import distance as dist
import csv
import numpy as np
from fer import FER
import threading
import time
import sys

Flag=False
MAX_FRAMES = 120 # Modify this to affect calibration period and amount of "lookback"
RECENT_FRAMES = int(MAX_FRAMES / 10) # Nodify to affect sensitivity to recent changes

EYE_BLINK_HEIGHT = .15 # Threshold may depend on relative face shape

SIGNIFICANT_BPM_CHANGE = 8

LIP_COMPRESSION_RATIO = .35 # From testing, ~universal

TELL_MAX_TTL = 30 # How long to display a finding, optionally set in args

TEXT_HEIGHT = 30

FACEMESH_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

EPOCH = time.time()


recording = None

tells = dict()

blinks = [False] * MAX_FRAMES
blinks2 = [False] * MAX_FRAMES # For mirroring

hand_on_face = [False] * MAX_FRAMES
hand_on_face2 = [False] * MAX_FRAMES # For mirroring

face_area_size = 0 # Relative size of face to total frame

hr_times = list(range(0, MAX_FRAMES))
hr_values = [400] * MAX_FRAMES
avg_bpms = [0] * MAX_FRAMES

gaze_values = [0] * MAX_FRAMES

emotion_detector = FER(mtcnn=True)
calculating_mood = False
mood = ''
frameNumber=0
meter = cv2.imread('meter.png')

# BPM chart
fig = None
ax = None
line = None
peakpts = None

def chart_setup():
	global fig, ax, line, peakpts

	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1) # 1st 1x1 subplot
	ax.set(ylim=(185, 200))
	line, = ax.plot(hr_times, hr_values, 'b-')
	peakpts, = ax.plot([], [], 'r+')

def decrement_tells(tells):
	for key, tell in tells.copy().items():
		if 'ttl' in tell:
			tell['ttl'] -= 1
			if tell['ttl'] <= 0:
				del tells[key]
	return tells
	
def main(INPUT, output_name = 'signals.csv'):
	global TELL_MAX_TTL
	global recording

	calibrated = False
	calibration_frames = 0
	with mp.solutions.face_mesh.FaceMesh(
			max_num_faces=1,
			refine_landmarks=True,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as face_mesh:
		with mp.solutions.hands.Hands(
				max_num_hands=2,
				min_detection_confidence=0.7) as hands:
			
				cap = cv2.VideoCapture(INPUT)
				fps = None
				if isinstance(INPUT, str) and INPUT.find('.') > -1: # From file
					fps = cap.get(cv2.CAP_PROP_FPS)
				else: # From device
					cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
					cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
					cap.set(cv2.CAP_PROP_FPS, 30)

				while cap.isOpened():
					success, image = cap.read()
					if not success: break
					calibration_frames += process(image, face_mesh, hands,INPUT, calibrated, fps, output_name = output_name)
					calibrated = (calibration_frames >= MAX_FRAMES)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break

				cap.release()

	cv2.destroyAllWindows()

def new_tell(result):
	global TELL_MAX_TTL
	
	# print (result)
	return {
		'text': result,
		'ttl': TELL_MAX_TTL
	}

def draw_on_frame(image, face_landmarks, hands_landmarks):
	mp.solutions.drawing_utils.draw_landmarks(
			image,
			face_landmarks,
			mp.solutions.face_mesh.FACEMESH_CONTOURS,
			landmark_drawing_spec=None,
			connection_drawing_spec=mp.solutions.drawing_styles
			.get_default_face_mesh_contours_style())
	mp.solutions.drawing_utils.draw_landmarks(
			image,
			face_landmarks,
			mp.solutions.face_mesh.FACEMESH_IRISES,
			landmark_drawing_spec=None,
			connection_drawing_spec=mp.solutions.drawing_styles
			.get_default_face_mesh_iris_connections_style())
	for hand_landmarks in (hands_landmarks or []):
		mp.solutions.drawing_utils.draw_landmarks(
				image,
				hand_landmarks,
				mp.solutions.hands.HAND_CONNECTIONS,
				mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
				mp.solutions.drawing_styles.get_default_hand_connections_style())

def add_text(image, tells, calibrated):
	global mood
	text_y = TEXT_HEIGHT
	if mood:
		write("Mood: {}".format(mood), image, int(.75 * image.shape[1]), TEXT_HEIGHT)
	if calibrated:
		for tell in tells.values():
			write(tell['text'], image, 10, text_y)
			text_y += TEXT_HEIGHT

def write(text, image, x, y):
	cv2.putText(img=image, text=text, org=(x, y),
		fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 0],
		lineType=cv2.LINE_AA, thickness=4)
	cv2.putText(img=image, text=text, org=(x, y),
		fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 255, 255],
		lineType=cv2.LINE_AA, thickness=2)

def get_aspect_ratio(top, bottom, right, left):
	height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
	width = dist.euclidean([right.x, right.y], [left.x, left.y])
	return height / width

def get_area(image, draw, topL, topR, bottomR, bottomL):
	topY = int((topR.y+topL.y)/2 * image.shape[0])
	botY = int((bottomR.y+bottomL.y)/2 * image.shape[0])
	leftX = int((topL.x+bottomL.x)/2 * image.shape[1])
	rightX = int((topR.x+bottomR.x)/2 * image.shape[1])

	if draw:
		image = cv2.circle(image, (leftX,topY), 2, (255,0,0), 2)
		image = cv2.circle(image, (leftX,botY), 2, (255,0,0), 2)
		image = cv2.circle(image, (rightX,topY), 2, (255,0,0), 2)
		image = cv2.circle(image, (rightX,botY), 2, (255,0,0), 2)

	return image[topY:botY, rightX:leftX]

def get_bpm_tells(cheekL, cheekR, fps, bpm_chart):
	global hr_times, hr_values, avg_bpms
	global ax, line, peakpts

	cheekLwithoutBlue = np.average(cheekL[:, :, 1:3])
	cheekRwithoutBlue = np.average(cheekR[:, :, 1:3])
	hr_values = hr_values[1:] + [cheekLwithoutBlue + cheekRwithoutBlue]
	# Calculating the average color value of the red and green channels for both cheeks
	
	if not fps:
		hr_times = hr_times[1:] + [time.time() - EPOCH]

	if bpm_chart:
		line.set_data(hr_times, hr_values)
		ax.relim()
		ax.autoscale()

	peaks, _ = find_peaks(hr_values,
		threshold=.1,
		distance=5,
		prominence=.5,
		wlen=10,
	)

	peak_times = [hr_times[i] for i in peaks]

	if bpm_chart:
		peakpts.set_data(peak_times, [hr_values[i] for i in peaks])

	bpms = 60 * np.diff(peak_times) / (fps or 1)
	bpms = bpms[(bpms > 50) & (bpms < 150)] # Filter to reasonable BPM range
	recent_bpms = bpms[(-3 * RECENT_FRAMES):] # HR slower signal than other tells

	recent_avg_bpm = 0
	bpm_display = "BPM: ..."
	if recent_bpms.size > 1:
		recent_avg_bpm = int(np.average(recent_bpms))
		bpm_display = "BPM: {} ({})".format(recent_avg_bpm, len(recent_bpms))

	avg_bpms = avg_bpms[1:] + [recent_avg_bpm]

	bpm_delta = 0
	bpm_change = ""

	if len(recent_bpms) > 2:
		all_bpms = list(filter(lambda bpm: bpm != '-', avg_bpms))
		all_avg_bpm = sum(all_bpms) / len(all_bpms)
		avg_recent_bpm = sum(recent_bpms) / len(recent_bpms)
		bpm_delta = avg_recent_bpm - all_avg_bpm

		if bpm_delta > SIGNIFICANT_BPM_CHANGE:
			bpm_change = "Heart rate increasing"
		elif bpm_delta < -SIGNIFICANT_BPM_CHANGE:
			bpm_change = "Heart rate decreasing"

	return recent_avg_bpm, bpm_change

def is_blinking(face):
	eyeR = [face[p] for p in [159, 145, 133, 33]]
	eyeR_ar = get_aspect_ratio(*eyeR)

	eyeL = [face[p] for p in [386, 374, 362, 263]]
	eyeL_ar = get_aspect_ratio(*eyeL)

	eyeA_ar = (eyeR_ar + eyeL_ar) / 2
	return eyeA_ar < EYE_BLINK_HEIGHT

def get_blink_tell(blinks):
	if sum(blinks[:RECENT_FRAMES]) < 3: # Not enough blinks for valid comparison
		return None

	recent_closed = 1.0 * sum(blinks[-RECENT_FRAMES:]) / RECENT_FRAMES
	avg_closed = 1.0 * sum(blinks) / MAX_FRAMES

	if recent_closed > (20 * avg_closed):
		return "Increased blinking"
	elif avg_closed >  (20 * recent_closed):
		return "Decreased blinking"
	else:
		return None

def check_hand_on_face(hands_landmarks, face):
	if hands_landmarks:
		face_landmarks = [face[p] for p in FACEMESH_FACE_OVAL]
		face_points = [[[p.x, p.y] for p in face_landmarks]]
		face_contours = np.array(face_points).astype(np.single)

		for hand_landmarks in hands_landmarks:
			hand = []
			for point in hand_landmarks.landmark:
				hand.append( (point.x, point.y) )

			for finger in [4, 8, 20]:
				overlap = cv2.pointPolygonTest(face_contours, hand[finger], False)
				if overlap != -1:
					return True
	return False

def get_avg_gaze(face):
	gaze_left = get_gaze(face, 476, 474, 263, 362)
	gaze_right = get_gaze(face, 471, 469, 33, 133)
	return round((gaze_left + gaze_right) / 2, 1)

def get_gaze(face, iris_L_side, iris_R_side, eye_L_corner, eye_R_corner):
	iris = (
		face[iris_L_side].x + face[iris_R_side].x,
		face[iris_L_side].y + face[iris_R_side].y,
	)
	eye_center = (
		face[eye_L_corner].x + face[eye_R_corner].x,
		face[eye_L_corner].y + face[eye_R_corner].y,
	)

	gaze_dist = dist.euclidean(iris, eye_center)
	eye_width = abs(face[eye_R_corner].x - face[eye_L_corner].x)
	gaze_relative = gaze_dist / eye_width

	if (eye_center[0] - iris[0]) < 0: # Flip along x for looking L vs R
		gaze_relative *= -1

	return gaze_relative

def detect_gaze_change(avg_gaze):
	global gaze_values

	gaze_values = gaze_values[1:] + [avg_gaze]
	gaze_relative_matches = 1.0 * gaze_values.count(avg_gaze) / MAX_FRAMES
	if gaze_relative_matches < .01: # Looking in a new direction
		return gaze_relative_matches
	return 0

def get_lip_ratio(face):
	return get_aspect_ratio(face[0], face[17], face[61], face[291])

def get_mood(image):
	global emotion_detector, calculating_mood, mood

	detected_mood, score = emotion_detector.top_emotion(image)
	calculating_mood = False
	if score and (score > .4 or detected_mood == 'neutral'):
		mood = detected_mood

def add_truth_meter(image, tell_count):
	width = image.shape[1]
	sm = int(width / 64)
	bg = int(width / 3.2)

	resized_meter = cv2.resize(meter, (bg,sm), interpolation=cv2.INTER_AREA)
	image[sm:(sm+sm), bg:(bg+bg), 0:3] = resized_meter[:, :, 0:3]

	if tell_count:
		tellX = bg + int(bg/4) * (tell_count - 1) # Adjust for always-on BPM
		cv2.rectangle(image, (tellX, int(.9*sm)), (tellX+int(sm/2), int(2.1*sm)), (0,0,0), 2)

def get_face_relative_area(face):
	face_width = abs(max(face[454].x, 0) - max(face[234].x, 0))
	face_height = abs(max(face[152].y, 0) - max(face[10].y, 0))
	return face_width * face_height

def find_face_and_hands(image_original, face_mesh, hands):
	image = image_original.copy()
	image.flags.writeable = False # Pass by reference to improve speed
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	faces = face_mesh.process(image)
	hands_landmarks = hands.process(image).multi_hand_landmarks

	face_landmarks = None
	if faces.multi_face_landmarks and len(faces.multi_face_landmarks) > 0:
		face_landmarks = faces.multi_face_landmarks[0] # Use first face found

	return face_landmarks, hands_landmarks

# Initialize lists to store the features of the last 10 frames
avg_bpms_list = []
blinking_list = []
hand_list = []
gaze_list = []
lips_list = []

def process(image, face_mesh, hands, video_name, calibrated=False, draw=False, bpm_chart=False, flip=False, fps=None, output_name = 'signals.csv'):
		global tells, calculating_mood, blinks, hand_on_face, face_area_size, frameNumber, avg_bpms_list, blinking_list, hand_list, gaze_list, lips_list
		if frameNumber % 50 == 0: print(frameNumber)
		frameNumber += 1

		tells = decrement_tells(tells)

		face_landmarks, hands_landmarks = find_face_and_hands(image, face_mesh, hands)
		if face_landmarks:
				face = face_landmarks.landmark
				face_area_size = get_face_relative_area(face)

				if not calculating_mood:
						start_mood_calculation_thread(image)
				
				cheekL = get_area(image, draw, face[449], face[350], face[429], face[280])
				cheekR = get_area(image, draw, face[121], face[229], face[50], face[209])

				avg_bpms, bpm_change = get_bpm_tells(cheekL, cheekR, fps, bpm_chart)
				tells['avg_bpms'] = new_tell(avg_bpms)
				
				update_blinks(face)
				update_hand_on_face(hands_landmarks, face)

				avg_gaze = get_avg_gaze(face)
				update_gaze(avg_gaze)

				update_lips(face)

				if bpm_chart:
						draw_bpm_chart()

				if draw:
						draw_on_frame(image, face_landmarks, hands_landmarks)
				
				store_frame_features(avg_bpms,face,video_name, output_name)

		return 1 if (face_landmarks and not calibrated) else 0

def start_mood_calculation_thread(image):
		global calculating_mood
		emothread = threading.Thread(target=get_mood, args=(image,))
		emothread.start()
		calculating_mood = True

def update_blinks(face):
		global tells, blinks
		blinks = blinks[1:] + [is_blinking(face)]
		recent_blink_tell = get_blink_tell(blinks)
		if recent_blink_tell:
				tells['blinking'] = new_tell(recent_blink_tell)

def update_hand_on_face(hands_landmarks, face):
		global tells, hand_on_face
		recent_hand_on_face = check_hand_on_face(hands_landmarks, face)
		hand_on_face = hand_on_face[1:] + [recent_hand_on_face]
		if recent_hand_on_face:
				tells['hand'] = new_tell("Hand covering face")

def update_gaze(avg_gaze):
		global tells
		if detect_gaze_change(avg_gaze):
				tells['gaze'] = new_tell("Change in gaze")

def update_lips(face):
		global tells
		if get_lip_ratio(face) < LIP_COMPRESSION_RATIO:
				tells['lips'] = new_tell("Lip compression")

def draw_bpm_chart():
		global fig
		fig.canvas.draw()
		fig.canvas.flush_events()

# Import the necessary modules

# Initialize a list to store data for batch writing
batch_data = []

# Define the store_frame_features function
def store_frame_features(avg_bpms, face, video_name, output_name):
		global batch_data

		# Append data for the current frame to the batch list
		batch_data.append({'avg_bpms': avg_bpms, 'blinking': get_blinking_value(), 'hand': np.mean(np.unique(hand_on_face)), 'gaze': get_avg_gaze(face), 'lips': get_lip_ratio(face), 'emotion': mood})

		# Check if batch size exceeds a certain threshold (e.g., 100 frames)
		if len(batch_data) >= 100:
				write_batch_to_csv(video_name, output_name)
				batch_data = []  # Reset the batch data after writing

# Define the write_batch_to_csv function
def write_batch_to_csv(video_name, output_name):
		global batch_data

		# Open the CSV file in append mode using a context manager
		with open(output_name, 'a', newline='') as csvfile:
				fieldnames = ['frameIdentifier', 'avg_bpms', 'blinking', 'hand', 'gaze', 'lips', 'emotion']
				writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

				# Iterate over the batch data and write it to the CSV file
				for frame_data in batch_data:
						frameIdentifier = f"{video_name}_{frameNumber+1}"
						frame_data['frameIdentifier'] = frameIdentifier
						writer.writerow(frame_data)


# Call write_batch_to_csv function to ensure any remaining data in the batch is written
def get_blinking_value():
		recent_blink_tell = get_blink_tell(blinks)
		if recent_blink_tell == "Increased blinking":
				return 1
		elif recent_blink_tell == "Decreased blinking":
				return 2
		else:
				return 0

def calculate_and_write_csv_data(video_name):
		global avg_bpms_list, blinking_list, hand_list, gaze_list, lips_list, frameNumber
		avg_bpms_avg = np.mean(avg_bpms_list)
		blinking_avg = get_blinking_value()
		hand_avg = np.mean(np.unique(hand_list))
		gaze_avg = np.mean(gaze_list)
		lips_avg = np.mean(lips_list)

		with open('Deceptive.csv', 'a', newline='') as csvfile:
				fieldnames = ['frameIdentifier', 'avg_bpms', 'blinking', 'hand', 'gaze', 'lips','emotion']
				writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

				frameIdentifier = f"{video_name}_{frameNumber+1}"

				writer.writerow({'frameIdentifier': str(frameIdentifier), 'avg_bpms': avg_bpms_avg, 'blinking': blinking_avg, 'hand': hand_avg, 'gaze': gaze_avg, 'lips': lips_avg,'emotion': mood})

		clear_lists()

def clear_lists():
		global avg_bpms_list, blinking_list, hand_list, gaze_list, lips_list
		avg_bpms_list = []
		blinking_list = []
		hand_list = []
		gaze_list = []
		lips_list = []

def flip_image(image):
		global flip
		if flip:
				image = cv2.flip(image, 1)

def mirror_compare(first, second, rate, less, more):
	if (rate * first) < second:
		return less
	elif first > (rate * second):
		return more
	return None

def get_blink_comparison(blinks1, blinks2):
	return mirror_compare(sum(blinks1), sum(blinks2), 1.8, "Blink less", "Blink more")

def get_hand_face_comparison(hand1, hand2):
	return mirror_compare(sum(hand1), sum(hand2), 2.1, "Stop touching face", "Touch face more")

def get_face_size_comparison(ratio1, ratio2):
	return mirror_compare(ratio1, ratio2, 1.5, "Too close", "Too far")

