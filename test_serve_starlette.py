import ray
from ray import serve
from fastapi import FastAPI
from starlette.routing import Route
import requests

app1 = FastAPI()

async def empty_post(request):
    return {"status": "empty_route_hit"}

app1.routes.append(Route("", endpoint=empty_post, methods=["POST"]))

@serve.deployment(route_prefix="/v1/images")
@serve.ingress(app1)
class D1:
    pass

ray.init(ignore_reinit_error=True)
serve.start()
serve.run(D1.bind(), name="d1")

print("/v1/images ->", requests.post("http://127.0.0.1:8000/v1/images", allow_redirects=False).text)
