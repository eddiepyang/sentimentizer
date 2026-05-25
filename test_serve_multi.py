import ray
from ray import serve
from fastapi import FastAPI
import requests

app1 = FastAPI()
@serve.deployment
@serve.ingress(app1)
class D1:
    @app1.post("/api/v1/foo")
    def foo(self): return "foo"

app2 = FastAPI()
@serve.deployment
@serve.ingress(app2)
class D2:
    @app2.post("/api/v1/bar")
    def bar(self): return "bar"

ray.init(ignore_reinit_error=True)
serve.start()
serve.run(D1.bind(), name="d1", route_prefix="/")
serve.run(D2.bind(), name="d2", route_prefix="/")

r1 = requests.post("http://127.0.0.1:8000/api/v1/foo", allow_redirects=False)
print("foo ->", r1.status_code)
r2 = requests.post("http://127.0.0.1:8000/api/v1/bar", allow_redirects=False)
print("bar ->", r2.status_code)
