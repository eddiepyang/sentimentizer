import ray
from ray import serve
from fastapi import FastAPI
import requests

app1 = FastAPI()
@serve.deployment
@serve.ingress(app1)
class D1:
    @app1.post("/v1/predict")
    def p(self): return "predict"

app2 = FastAPI()
@serve.deployment
@serve.ingress(app2)
class D2:
    @app2.post("/images")
    def img(self): return "images"

ray.init(ignore_reinit_error=True)
serve.start()
serve.run(D1.bind(), name="d1", route_prefix="/")
serve.run(D2.bind(), name="d2", route_prefix="/v1")

print("/v1/predict ->", requests.post("http://127.0.0.1:8000/v1/predict").status_code)
print("/v1/images ->", requests.post("http://127.0.0.1:8000/v1/images").status_code)
