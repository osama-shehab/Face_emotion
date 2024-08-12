import unittest
import numpy as np
import cv2
import os
from face_emotion_lib import face_recognition, emotion_detection, utils

class TestFaceEmotionLib(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup any state before tests run
        cls.test_dir = 'test_images'
        cls.encoding_file = 'test_encodings.pkl'
        
        # Create a directory with some test images
        if not os.path.exists(cls.test_dir):
            os.mkdir(cls.test_dir)
        cls.create_test_images(cls.test_dir)
    
    @classmethod
    def tearDownClass(cls):
        # Cleanup after tests are done
        if os.path.exists(cls.test_dir):
            for file in os.listdir(cls.test_dir):
                os.remove(os.path.join(cls.test_dir, file))
            os.rmdir(cls.test_dir)
        if os.path.exists(cls.encoding_file):
            os.remove(cls.encoding_file)

    @staticmethod
    def create_test_images(directory):
        # Create a simple test image with a solid color (black)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(directory, 'test_face.jpg'), image)

    def test_load_known_faces(self):
        encodings, names = face_recognition.load_known_faces(self.test_dir, self.encoding_file)
        self.assertGreater(len(encodings), 0, "No face encodings were loaded")
        self.assertGreater(len(names), 0, "No face names were loaded")

    def test_detect_emotion(self):
        # Since emotion detection requires a valid image file and DeepFace, 
        # this is a placeholder test. You'll need to mock DeepFace.analyze in real tests.
        frame = os.path.join(self.test_dir, 'test_face.jpg')
        emotion = emotion_detection.detect_emotion(frame)
        # Emotion detection might be hard to test without actual data, so check for type or structure
        self.assertIsInstance(emotion, str, "Emotion detection did not return a string")

    def test_create_black_screen_with_details(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fps = 30.0
        message = "Test Message"
        black_screen = utils.create_black_screen_with_details(message, frame, fps)
        self.assertEqual(black_screen.shape, frame.shape, "Black screen shape does not match frame shape")
        self.assertIsInstance(black_screen, np.ndarray, "Black screen is not a numpy array")

    def test_log_details(self):
        log_file = 'test_log.txt'
        identity = 'Test Name'
        emotion = 'Happy'
        utils.log_details(identity, emotion, log_file)
        
        # Check if the log file was created and contains the expected data
        self.assertTrue(os.path.exists(log_file), "Log file was not created")
        with open(log_file, 'r') as file:
            content = file.read()
            self.assertIn(identity, content, "Identity not found in log file")
            self.assertIn(emotion, content, "Emotion not found in log file")

        # Clean up
        os.remove(log_file)

if __name__ == '__main__':
    unittest.main()
