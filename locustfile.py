from locust import HttpUser, task, between

class ChurnApiUser(HttpUser):
    # wait between 1 and 3 seconds between requests (simulates real users)
    wait_time = between(1, 3)

    @task(10)
    def predict(self):
        # your sample payload
        payload = {
            "CreditScore":      650,
            "Geography":        "France",
            "Gender":           "Male",
            "Age":               40,
            "Tenure":             3,
            "Balance":        60000.0,
            "NumOfProducts":      2,
            "HasCrCard":          1,
            "IsActiveMember":     1,
            "EstimatedSalary": 50000.0
        }
        # hit the /predict endpoint
        self.client.post("/predict", json=payload)
