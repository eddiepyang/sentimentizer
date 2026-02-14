# Kubernetes Deployment for Sentimentizer

This directory contains Kubernetes manifests for deploying the sentimentizer Ray Serve application.

## Prerequisites

1. Build the Docker image:
```bash
docker build -t sentimentizer:latest .
```

2. If using a remote Kubernetes cluster, push the image to your container registry:
```bash
docker tag sentimentizer:latest <your-registry>/sentimentizer:latest
docker push <your-registry>/sentimentizer:latest
```

Then update the `image` field in [deployment.yaml](deployment.yaml).

## Deployment

Apply all manifests:
```bash
kubectl apply -f k8s/
```

Or apply individually:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Verify Deployment

Check pod status:
```bash
kubectl get pods -l app=sentimentizer
```

Check service:
```bash
kubectl get svc sentimentizer
```

View logs:
```bash
kubectl logs -l app=sentimentizer -f
```

## Access the Service

### From within the cluster:
```bash
curl -X POST http://sentimentizer:8000 \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a great product!"}'
```

### Port-forward for local testing:
```bash
kubectl port-forward svc/sentimentizer 8000:8000
```

Then use the Go client:
```bash
./sentiment -host http://localhost:8000 -text "This is amazing!"
```

### Access Ray Dashboard:
```bash
kubectl port-forward svc/sentimentizer 8265:8265
```

Then open http://localhost:8265 in your browser.

## Expose Externally (Optional)

If you need external access, you can:

1. **NodePort** - Add to [service.yaml](service.yaml):
```yaml
spec:
  type: NodePort
```

2. **LoadBalancer** - Change service type:
```yaml
spec:
  type: LoadBalancer
```

3. **Ingress** - Create an Ingress resource (requires Ingress controller):
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sentimentizer
spec:
  rules:
  - host: sentimentizer.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sentimentizer
            port:
              number: 8000
```

## Resource Configuration

The deployment is configured with:
- **Replicas**: 1 (single node)
- **CPU Request**: 500m (0.5 cores)
- **CPU Limit**: 2000m (2 cores)
- **Memory Request**: 512Mi
- **Memory Limit**: 2Gi

Adjust these in [deployment.yaml](deployment.yaml) based on your workload.

## Cleanup

Remove all resources:
```bash
kubectl delete -f k8s/
```
