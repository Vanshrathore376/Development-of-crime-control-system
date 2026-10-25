from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import os
import json
import uuid
import base64

from services.face_engine import FaceEngine


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
DATABASE_FILE = os.path.join(
    BASE_DIR,
    "database",
    "persons.json"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(
    os.path.dirname(DATABASE_FILE),
    exist_ok=True
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# -----------------------------
# Load AI engine
# -----------------------------

engine = FaceEngine()


# -----------------------------
# Database functions
# -----------------------------

def load_database():

    if not os.path.exists(DATABASE_FILE):
        return []

    with open(
        DATABASE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_database(data):

    with open(
        DATABASE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# -----------------------------
# Utility
# -----------------------------

def allowed_file(filename):

    allowed = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in allowed


def embedding_to_list(feature):

    return feature.flatten().astype(
        float
    ).tolist()


def list_to_embedding(values):

    return np.array(
        values,
        dtype=np.float32
    ).reshape(1, -1)


def image_to_base64(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image
    )

    if not success:
        return None

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")

    return "data:image/jpeg;base64," + encoded


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# -----------------------------
# Add person
# -----------------------------

@app.route(
    "/api/person/add",
    methods=["POST"]
)
def add_person():

    name = request.form.get(
        "name",
        ""
    ).strip()

    case_id = request.form.get(
        "case_id",
        ""
    ).strip()

    file = request.files.get(
        "image"
    )

    if not name:
        return jsonify({
            "success": False,
            "message": "Person name is required."
        }), 400

    if not file:
        return jsonify({
            "success": False,
            "message": "Reference image is required."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": "Invalid image format."
        }), 400

    filename = secure_filename(
        file.filename
    )

    unique_name = (
        str(uuid.uuid4())
        + "_"
        + filename
    )

    path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    file.save(path)

    image = cv2.imread(path)

    if image is None:
        return jsonify({
            "success": False,
            "message": "Unable to read image."
        }), 400

    face, feature = engine.extract_first_face(
        image
    )

    if feature is None:
        return jsonify({
            "success": False,
            "message": "No face detected in reference image."
        }), 400

    database = load_database()

    person = {
        "id": str(uuid.uuid4()),
        "name": name,
        "case_id": case_id,
        "image": unique_name,
        "encoding": embedding_to_list(feature)
    }

    database.append(person)

    save_database(database)

    return jsonify({
        "success": True,
        "message": "Person added successfully.",
        "person_id": person["id"],
        "name": name
    })


# -----------------------------
# List persons
# -----------------------------

@app.route("/api/persons")
def persons():

    database = load_database()

    output = []

    for person in database:

        output.append({
            "id": person["id"],
            "name": person["name"],
            "case_id": person["case_id"]
        })

    return jsonify(output)


# -----------------------------
# Identify person in group image
# -----------------------------

@app.route(
    "/api/identify",
    methods=["POST"]
)
def identify():

    person_id = request.form.get(
        "person_id"
    )

    file = request.files.get(
        "group_image"
    )

    if not person_id:
        return jsonify({
            "success": False,
            "message": "Select a person."
        }), 400

    if not file:
        return jsonify({
            "success": False,
            "message": "Group image is required."
        }), 400

    database = load_database()

    selected_person = None

    for person in database:

        if person["id"] == person_id:
            selected_person = person
            break

    if selected_person is None:

        return jsonify({
            "success": False,
            "message": "Person not found."
        }), 404

    filename = secure_filename(
        file.filename
    )

    unique_name = (
        str(uuid.uuid4())
        + "_"
        + filename
    )

    group_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    file.save(group_path)

    image = cv2.imread(
        group_path
    )

    if image is None:

        return jsonify({
            "success": False,
            "message": "Unable to read group image."
        }), 400

    reference_feature = list_to_embedding(
        selected_person["encoding"]
    )

    results = engine.recognize_group(
        reference_feature,
        image
    )

    # Remove feature from JSON response
    clean_results = []

    best_match = None

    for result in results:

        clean_result = {
            "face_index": result["face_index"],
            "x": result["x"],
            "y": result["y"],
            "width": result["width"],
            "height": result["height"],
            "similarity": round(
                result["similarity"],
                4
            ),
            "distance": round(
                result["distance"],
                4
            ),
            "matched": result["matched"]
        }

        clean_results.append(
            clean_result
        )

        if result["matched"]:

            if (
                best_match is None
                or result["similarity"]
                > best_match["similarity"]
            ):

                best_match = result

    # Draw boxes
    for result in results:

        x = result["x"]
        y = result["y"]
        w = result["width"]
        h = result["height"]

        if result["matched"]:

            color = (0, 255, 0)
            label = (
                f"MATCH "
                f"{result['similarity']:.2f}"
            )

        else:

            color = (0, 0, 255)
            label = (
                f"NO MATCH "
                f"{result['similarity']:.2f}"
            )

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        cv2.putText(
            image,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    result_filename = (
        str(uuid.uuid4())
        + "_result.jpg"
    )

    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )

    cv2.imwrite(
        result_path,
        image
    )

    return jsonify({
        "success": True,
        "person": {
            "name": selected_person["name"],
            "case_id": selected_person["case_id"]
        },
        "faces_detected": len(results),
        "matched": best_match is not None,
        "best_similarity": (
            round(
                best_match["similarity"],
                4
            )
            if best_match
            else 0
        ),
        "best_distance": (
            round(
                best_match["distance"],
                4
            )
            if best_match
            else None
        ),
        "results": clean_results,
        "result_image": image_to_base64(image)
    })


# -----------------------------
# Health check
# -----------------------------

@app.route("/api/health")
def health():

    return jsonify({
        "status": "running",
        "system": "Crime Control Face Recognition",
        "python": "3.13"
    })


# -----------------------------
# Start server
# -----------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
