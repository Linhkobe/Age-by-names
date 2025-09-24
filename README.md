# Gender by Name API 

## 1. About the project 
A FastAPI + Pytorch web service to predict the age of a person from their first name (the neural network model was trained with french firt names dataset), packaged with Docker and deployed on Google Cloud Run. 

Here is the deployed link (would take few moments to open) : https://age-by-names-561199547702.europe-west1.run.app/ 

## 2. Features
- Predict age from a given first name
- Uses PyTorch regression model + preprocessing encoders (scalers, target encoder)
- RESTful API built with FastAPI
- Simple HTML front-end for quick testting
- Dockerized for portability
- Production deployment via Google Cloud Run

## 3. Project structure 
```
gender_by_name/
├── app/                  # FastAPI app
│   ├── main.py           # Entrypoint
│   ├── service.py        # Model + encoding logic
│   ├── templates/        # index.html (frontend)
│   └── static/           # CSS, favicon, etc.
├── model/
│   └── prod/
│       ├── torch_regression_model.pkl # Trained with PyTorch model 
│       ├── prenom_age_encoder.pkl # Encoder for forst names
│       ├── target_encoder_age.pkl # Target encoding
│       └── sacler_target.pkl # Scaler for regression output
├── requirements.txt
├── Dockerfile
└── README.md
```

## 4. Steps for local development 
1. Clone repository

```
git clone https://github.com/Linhkobe/Age-by-names.git
cd age_by_names
```

2. Install dependencies (Python 3.10+)

 ```
 python3 -m venv venv
 source venv/bin/activate
 pip install -r requirements.txt
 ```

 3. Run API locally 

```
uvicorn app.main:app --reload --port 8001
```

4. Open http://localhost:8001 on browser to test

## 5. Docker

* Install Docker desktop as prerequisite

1. Build image

```
docker build -t age-by-names:v1 .
```

2. Run container
```
docker run --rm -p 8001:8001 age-by-names:v1
```

3. Test in browser with http://localhost:8001


## 6. Deployment to Google Cloud Run

* Create a project and get its ID via https://cloud.google.com/?hl=en as prerequisite 

1. Enable APIs

```
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

2. Create artifact registry repo

```
gcloud artifacts repositories create gender-repo \
  --repository-format=docker
  --location=europe-west1 \
  --description="Docker repo for Age-by-names"
```

3. Build & push image

```
gcloud auth configure-docker europe-west1-docker.pkg.dev
docker build -t europe-west1-docker.pkg.dev/<PROJECT_ID>/age-repo/age-by-names:v1 .
docker push europe-west1-docker.pkg.dev/<PROJECT_ID>/age-repo/age-by-names:v1
```

4. Deploy to Google Cloud Run

```
gcloud run deploy age-by-names \
  --image europe-west1-docker.pkg.dev/<PROJECT_ID>/age-repo/age-by-names:v1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Go \
  --timeout 600 \
  --set-env-vars MODEL_DIR=/app/model/prod,VERSION=v1
```


