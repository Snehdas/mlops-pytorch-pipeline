# MLOps PyTorch Pipeline with Kubernetes

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for training and serving a PyTorch ResNet18 image classification model using Docker and Kubernetes.

The pipeline demonstrates:

* Containerized model training
* Kubernetes-based training using a `Job`
* Model serving using a Kubernetes `Deployment`
* Kubernetes `Service` for inference access
* Horizontal Pod Autoscaling (HPA)
* Health-check and prediction endpoints
* Git/GitHub branch and pull-request workflow
* End-to-end validation of the ML workflow

The model is trained on the CIFAR-10 dataset and exposed through an HTTP inference service.

---

## 2. Architecture

```text
                         ┌──────────────────────┐
                         │      Developer       │
                         │   Git / GitHub PRs   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       main           │
                         │      Branch          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │       Kubernetes Cluster    │
                     │                             │
                     │  ┌───────────────────────┐  │
                     │  │   Training Job        │  │
                     │  │                       │  │
                     │  │  PyTorch + ResNet18   │  │
                     │  │       CIFAR-10        │  │
                     │  └───────────┬───────────┘  │
                     │              │              │
                     │              ▼              │
                     │     ┌─────────────────┐     │
                     │     │   Trained Model │     │
                     │     └────────┬────────┘     │
                     │              │              │
                     │              ▼              │
                     │  ┌────────────────────────┐ │
                     │  │ Model Serving          │ │
                     │  │ Kubernetes Deployment  │ │
                     │  │                        │ │
                     │  │ ┌──────┐    ┌──────┐  │ │
                     │  │ │Pod 1 │    │Pod 2 │  │ │
                     │  │ │2/2   │    │2/2   │  │ │
                     │  │ └───┬──┘    └───┬──┘  │ │
                     │  └──────┼───────────┼─────┘ │
                     │         │           │       │
                     │         └─────┬─────┘       │
                     │               ▼             │
                     │      ┌────────────────┐     │
                     │      │ Kubernetes     │     │
                     │      │ Service        │     │
                     │      └───────┬────────┘     │
                     │              │              │
                     │              ▼              │
                     │       /health /predict      │
                     │              │              │
                     │              ▼              │
                     │      ┌────────────────┐      │
                     │      │      HPA       │      │
                     │      │ CPU Target 70%│      │
                     │      │ Min: 2        │      │
                     │      │ Max: 5         │      │
                     │      └────────────────┘      │
                     └─────────────────────────────┘
```

### Workflow

```text
Code
  │
  ▼
Feature Branch
  │
  ▼
PR #6
  │
  ▼
develop
  │
  ▼
PR #7
  │
  ▼
main
  │
  ▼
Kubernetes
  │
  ├── Training Job
  │       │
  │       ▼
  │   Trained Model
  │       │
  │       ▼
  └── Model Serving
          │
          ├── /health
          ├── /predict
          │
          └── HPA
              └── 2–5 replicas
```

---

## 3. Project Structure

```text
mlops-pytorch-pipeline/
│
├── app/
│   └── ...
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
│
├── Dockerfile
├── requirements.txt
├── README.md
└── ...
```

---

## 4. Prerequisites

Install the following before running the project:

* Docker
* Kubernetes
* `kubectl`
* A local Kubernetes cluster such as Docker Desktop Kubernetes or Minikube
* Git

Verify the installations:

```bash
docker --version
kubectl version --client
git --version
```

Verify that Kubernetes is running:

```bash
kubectl cluster-info
```

---

## 5. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd mlops-pytorch-pipeline
```

---

## 6. Build the Docker Image

Build the project image:

```bash
docker build -t mlops-pytorch:latest .
```

Verify the image:

```bash
docker images
```

If using a local Kubernetes cluster such as Docker Desktop, the locally built image can be used directly depending on the cluster configuration.

---

## 7. Deploy the Kubernetes Resources

Create the project namespace:

```bash
kubectl apply -f k8s/namespace.yaml
```

Apply the configuration:

```bash
kubectl apply -f k8s/configmap.yaml
```

Start the model-training Job:

```bash
kubectl apply -f k8s/training-job.yaml
```

Check the training Job:

```bash
kubectl get jobs -n ml-training
```

Check the training pod:

```bash
kubectl get pods -n ml-training
```

View training logs:

```bash
kubectl logs job/model-training -n ml-training
```

Wait until the training Job completes successfully before starting model serving.

---

## 8. Deploy Model Serving

After successful training, deploy the model-serving application:

```bash
kubectl apply -f k8s/serving-deployment.yaml
```

Apply the Kubernetes Service:

```bash
kubectl apply -f k8s/serving-service.yaml
```

Apply the Horizontal Pod Autoscaler:

```bash
kubectl apply -f k8s/hpa.yaml
```

Check the deployment:

```bash
kubectl get deployment model-serving -n ml-training
```

Expected result:

```text
NAME            READY   UP-TO-DATE   AVAILABLE
model-serving   2/2     2            2
```

---

## 9. Verify HPA

Check the HPA:

```bash
kubectl get hpa -n ml-training
```

The configured autoscaling policy uses:

```text
Minimum replicas: 2
Maximum replicas: 5
CPU target:       70%
```

The HPA allows Kubernetes to increase or decrease the number of serving replicas according to CPU utilization.

---

## 10. Test the Health Endpoint

Port-forward the serving service:

```bash
kubectl port-forward service/model-serving 8000:8000 -n ml-training
```

Open another terminal and test:

```bash
curl http://localhost:8000/health
```

The endpoint should report that the model-serving application is healthy.

---

## 11. Test Prediction

The `/predict` endpoint accepts an image and returns the model prediction.

Example:

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/image.png"
```

A successful response confirms that:

1. The serving application is running.
2. The trained model is loaded.
3. The inference endpoint is accessible.
4. The model can successfully generate predictions.

---

## 12. Useful Kubernetes Commands

View all resources:

```bash
kubectl get all -n ml-training
```

View pods:

```bash
kubectl get pods -n ml-training
```

View deployments:

```bash
kubectl get deployments -n ml-training
```

View services:

```bash
kubectl get services -n ml-training
```

View HPA:

```bash
kubectl get hpa -n ml-training
```

View pod logs:

```bash
kubectl logs <POD_NAME> -n ml-training
```

Describe a resource:

```bash
kubectl describe pod <POD_NAME> -n ml-training
```

---

## 13. End-to-End Validation

The complete workflow was validated using the following checks:

| Component    | Validation                                             |
| ------------ | ------------------------------------------------------ |
| Training     | Kubernetes training Job completed successfully         |
| Serving      | Deployment reached 2/2 available replicas              |
| HPA          | CPU target configured at 70%, 2–5 replicas             |
| Health       | `/health` endpoint returned a healthy service response |
| Prediction   | `/predict` successfully returned a CIFAR-10 prediction |
| Git workflow | Feature branch → `develop` → `main`                    |
| PR #6        | Feature changes merged into `develop`                  |
| PR #7        | `develop` merged into `main`                           |

---

## 14. Git Workflow

Development followed a feature-branch workflow:

```text
feature/kubernetes-serving
          │
          │ Pull Request #6
          ▼
       develop
          │
          │ Pull Request #7
          ▼
         main
```

This workflow provided controlled integration of changes and demonstrated the use of Git branches and pull requests in an MLOps development process.

---

## 15. Troubleshooting

### Check whether the training Job completed

```bash
kubectl get jobs -n ml-training
```

If the Job has not completed, inspect its logs:

```bash
kubectl logs job/model-training -n ml-training
```

### Check serving pod status

```bash
kubectl get pods -n ml-training
```

If a pod is not ready:

```bash
kubectl describe pod <POD_NAME> -n ml-training
```

Then inspect its logs:

```bash
kubectl logs <POD_NAME> -n ml-training
```

### Check deployment availability

```bash
kubectl get deployment model-serving -n ml-training
```

### Check HPA

```bash
kubectl get hpa -n ml-training
```

---

## 16. Cleanup

To remove the Kubernetes resources:

```bash
kubectl delete namespace ml-training
```

To remove the Docker image:

```bash
docker rmi mlops-pytorch:latest
```

---

## 17. Conclusion

This project demonstrates a complete MLOps workflow in which a PyTorch model is containerized, trained using a Kubernetes Job, deployed as a scalable inference service, and validated through health and prediction endpoints. Kubernetes HPA provides automatic scaling based on CPU utilization, while Git branches and pull requests provide controlled version management and integration.

The successful training, deployment, health-check, prediction, autoscaling configuration, and GitHub PR workflow demonstrate an end-to-end production-oriented machine learning deployment process.
