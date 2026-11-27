
import sys
import os
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.url_feature_extraction import URLFeatureExtractor


# -------------------- ENVIRONMENT --------------------

load_dotenv()

# Phishing predictions use the saved local model and do not require MongoDB.
# MongoDB is only needed by the separate model-training/data-ingestion pipeline.


# -------------------- FASTAPI APPLICATION --------------------

app = FastAPI(
    title="Network Security - Phishing Detection",
    description="Machine Learning Based Phishing Detection System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")


# -------------------- URL FEATURE EXTRACTOR --------------------

try:
    url_extractor = URLFeatureExtractor()
    print("URLFeatureExtractor initialized successfully")

except Exception as e:
    print(f"Warning: URLFeatureExtractor initialization failed: {e}")
    url_extractor = None


# -------------------- REGISTERED ROUTES --------------------

print("\nRoutes registered:")
print("GET  /")
print("GET  /health")
print("GET  /train")
print("GET  /train-run")
print("POST /predict")
print("POST /predict-url")
print("GET  /url-result")
print("\nFastAPI application ready!\n")


# -------------------- HEALTH CHECK --------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "url_extractor": url_extractor is not None
    }


# -------------------- HOME PAGE --------------------

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )


# -------------------- TRAIN PAGE --------------------

@app.get("/train")
def train_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="train_status.html",
        context={
            "request": request
        }
    )


# -------------------- TRAIN MODEL --------------------

@app.get("/train-run")
def train_run():

    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

        return {
            "status": "success",
            "message": "Training completed successfully"
        }

    except Exception as e:

        print(f"Training error: {e}")

        return JSONResponse(
            content={
                "status": "error",
                "message": str(e)
            },
            status_code=500
        )


# -------------------- CSV PREDICTION --------------------

@app.post("/predict")
def predict_route(
    request: Request,
    file: UploadFile = File(...)
):

    try:
        # Read uploaded CSV
        df = pd.read_csv(file.file)

        # Load trained model
        model = load_object("final_model/model.pkl")

        # Generate predictions
        y_pred = model.predict(df)

        # Add prediction results
        df["predicted_column"] = y_pred

        df["predicted_column"] = df["predicted_column"].replace({
            1: "Legitimate",
            0: "Phishing"
        })

        # Save prediction output
        os.makedirs("prediction_output", exist_ok=True)

        df.to_csv(
            "prediction_output/output.csv",
            index=False
        )

        # Convert dataframe to HTML
        table_html = df.to_html(
            classes="table table-striped",
            index=False
        )

        # Render result template
        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={
                "request": request,
                "table": table_html
            }
        )

    except Exception as e:

        raise NetworkSecurityException(e, sys)

    finally:
        file.file.close()


# -------------------- SINGLE URL PREDICTION --------------------

@app.post("/predict-url")
async def predict_url_post(request: Request):

    try:
        print("Received URL prediction request")

        # Read JSON request
        data = await request.json()

        url = data.get("url", "").strip()

        print(f"URL: {url}")

        # Validate URL
        if not url:

            return JSONResponse(
                content={
                    "success": False,
                    "error": "URL cannot be empty"
                },
                status_code=400
            )

        # Check feature extractor
        if url_extractor is None:

            return JSONResponse(
                content={
                    "success": False,
                    "error": "URL extractor not available"
                },
                status_code=500
            )

        # Extract URL features
        print("Extracting URL features...")

        try:
            features_df = url_extractor.extract_features_dataframe(url)

            print(f"Features extracted: {features_df.shape}")

            # Calculate heuristic phishing score
            phishing_score = url_extractor.calculate_phishing_score(url)

            print(f"Phishing score: {phishing_score:.1f}/100")

        except Exception as extract_error:

            print(f"Feature extraction error: {extract_error}")

            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Feature extraction failed: {str(extract_error)}"
                },
                status_code=400
            )

        # Load trained model
        print("Loading trained model...")

        try:
            model = load_object("final_model/model.pkl")

            print("Model loaded successfully")

        except Exception as model_error:

            print(f"Model loading error: {model_error}")

            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Model loading failed: {str(model_error)}"
                },
                status_code=500
            )

        # Make prediction
        print("Making prediction...")

        try:
            y_pred = model.predict(features_df)

            # Calculate prediction confidence
            try:
                y_proba = model.predict_proba(features_df)

                confidence = float(max(y_proba[0])) * 100

            except (AttributeError, ValueError, TypeError):
                confidence = 100.0

            # Convert prediction label
            prediction_label = (
                "Legitimate" if y_pred[0] == 1 else "Phishing"
            )

            print(
                f"Prediction: {prediction_label} "
                f"({confidence:.2f}%)"
            )

            return JSONResponse(
                content={
                    "success": True,
                    "url": url,
                    "prediction": prediction_label,
                    "confidence": round(confidence, 2),
                    "phishing_score": round(phishing_score, 1)
                }
            )

        except Exception as pred_error:

            print(f"Prediction error: {pred_error}")

            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Prediction failed: {str(pred_error)}"
                },
                status_code=500
            )

    except Exception as e:

        print(f"Request error: {e}")

        return JSONResponse(
            content={
                "success": False,
                "error": f"Request error: {str(e)}"
            },
            status_code=500
        )


# -------------------- URL RESULT PAGE --------------------

@app.get("/url-result")
def url_result(
    request: Request,
    url: str = "",
    prediction: str = "",
    confidence: float = 0,
    phishing_score: float = 0,
    error: str = None
):

    return templates.TemplateResponse(
        request=request,
        name="url_result.html",
        context={
            "request": request,
            "url": url,
            "prediction": prediction,
            "confidence": confidence,
            "phishing_score": phishing_score,
            "error": error
        }
    )


# -------------------- RUN APPLICATION --------------------

if __name__ == "__main__":

    app_run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
