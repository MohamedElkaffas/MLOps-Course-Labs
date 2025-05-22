from locust import HttpUser, task, between

class ChurnUser(HttpUser):
    wait_time = between(1, 2)  # random 1–2s between requests

    @task
    def predict(self):
        payload = {
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Male",
            "Age": 40,
            "Tenure": 3,
            "Balance": 60000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000.0
        }
        headers = {"Content-Type": "application/json"}
        # Note the use of json=payload — this automatically
        # serializes to JSON and sets the header.
        self.client.post("/predict", json=payload, headers=headers)
