import src.features as features
import sys
sys.modules["features"] = features

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from src.predict import predict_student, InvalidStudentInput
from backend.database import get_connection

app = FastAPI(
    title="Student Performance Prediction API",
    description="API for Student Performance Prediction",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    student: Dict[str, Any]


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running!"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # 1. Make prediction using ML model
        result = predict_student(
            request.student,
            return_details=True
        )

        # 2. Connect to MySQL
        connection = get_connection()
        cursor = connection.cursor()

        # 3. Save prediction in database
        query = """
        INSERT INTO predictions
        (age, study_time, failures, absences, predicted_grade)
        VALUES (%s, %s, %s, %s, %s)
        """

        values = (
            request.student["age"],
            request.student["studytime"],
            request.student["failures"],
            request.student["absences"],
            result["predicted_grade"]
        )

        cursor.execute(query, values)
        connection.commit()

        # 4. Close database connection
        cursor.close()
        connection.close()

        # 5. Return prediction
        return result

    except InvalidStudentInput as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Database or prediction error: {error}"
        )

@app.get("/predictions")
def get_predictions():
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT
            id,
            age,
            study_time,
            failures,
            absences,
            predicted_grade,
            created_at
        FROM predictions
        ORDER BY id DESC
        """

        cursor.execute(query)
        predictions = cursor.fetchall()

        cursor.close()
        connection.close()

        return {
            "count": len(predictions),
            "predictions": predictions
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {error}"
        )


    