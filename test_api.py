
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Student Performance Prediction API is running!"
    }


def test_prediction_endpoint():
    student_data = {
        "Hours_Studied": 5,
        "Attendance": 90,
        "Sleep_Hours": 7,
        "Previous_Scores": 75,
        "Tutoring_Sessions": 1,
        "Physical_Activity": 3,
        "Parental_Involvement": "High",
        "Access_to_Resources": "High",
        "Extracurricular_Activities": "Yes",
        "Motivation_Level": "High",
        "Internet_Access": "Yes",
        "Family_Income": "Medium",
        "Teacher_Quality": "High",
        "School_Type": "Public",
        "Peer_Influence": "Positive",
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "College",
        "Distance_from_Home": "Near",
        "Gender": "Male"
    }

    response = client.post("/predict", json=student_data)

    assert response.status_code == 200

    result = response.json()
    assert isinstance(result, dict)
    assert result

    # At least one response value should be a numeric prediction.
    assert any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in result.values()
    )
