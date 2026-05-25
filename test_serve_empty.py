import ray
from ray import serve
from fastapi import FastAPI
import requests

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class Deployment:
    @app.post("")
    def handle_empty(self):
        return "empty"
        
    @app.post("/")
    def handle_slash(self):
        return "slash"

ray.init(ignore_reinit_error=True)
serve.start()
serve.run(Deployment.bind(), name="test", route_prefix="/v1/images")

r1 = requests.post("http://127.0.0.1:8000/v1/images", allow_redirects=False)
print("/v1/images ->", r1.status_code, r1.text)

r2 = requests.post("http://127.0.0.1:8000/v1/images/", allow_redirects=False)
print("/v1/images/ ->", r2.status_code, r2.text)
