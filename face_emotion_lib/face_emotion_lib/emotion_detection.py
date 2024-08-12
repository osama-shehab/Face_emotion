from deepface import DeepFace

def detect_emotion(frame):
    try:
        emotion_results = DeepFace.analyze(img_path=frame, actions=['emotion'])
        return emotion_results[0]['dominant_emotion']
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return None

