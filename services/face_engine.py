import cv2
import numpy as np
import os


class FaceEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        yunet_path = os.path.join(
            base_dir,
            "models",
            "face_detection_yunet_2023mar.onnx"
        )

        sface_path = os.path.join(
            base_dir,
            "models",
            "face_recognition_sface_2021dec.onnx"
        )

        if not os.path.exists(yunet_path):
            raise FileNotFoundError(
                f"YuNet model not found: {yunet_path}"
            )

        if not os.path.exists(sface_path):
            raise FileNotFoundError(
                f"SFace model not found: {sface_path}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            yunet_path,
            "",
            (320, 320),
            0.7,
            0.3,
            5000
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            sface_path,
            ""
        )

    def detect_faces(self, image):

        height, width = image.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        return faces

    def get_embedding(self, image, face):

        aligned_face = self.recognizer.alignCrop(
            image,
            face
        )

        feature = self.recognizer.feature(
            aligned_face
        )

        return feature

    def cosine_similarity(self, feature1, feature2):

        score = self.recognizer.match(
            feature1,
            feature2,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        return float(score)

    def euclidean_distance(self, feature1, feature2):

        distance = self.recognizer.match(
            feature1,
            feature2,
            cv2.FaceRecognizerSF_FR_NORM_L2
        )

        return float(distance)

    def extract_first_face(self, image):

        faces = self.detect_faces(image)

        if len(faces) == 0:
            return None, None

        face = faces[0]

        feature = self.get_embedding(
            image,
            face
        )

        return face, feature

    def recognize_group(
        self,
        reference_feature,
        group_image,
        similarity_threshold=0.40,
        distance_threshold=1.10
    ):

        faces = self.detect_faces(group_image)

        results = []

        for index, face in enumerate(faces):

            feature = self.get_embedding(
                group_image,
                face
            )

            similarity = self.cosine_similarity(
                reference_feature,
                feature
            )

            distance = self.euclidean_distance(
                reference_feature,
                feature
            )

            # SFace official examples use cosine similarity
            # and L2 distance as matching measures.
            matched = (
                similarity >= similarity_threshold
                and distance <= distance_threshold
            )

            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])

            results.append({
                "face_index": index + 1,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "similarity": similarity,
                "distance": distance,
                "matched": matched,
                "feature": feature
            })

        return results
