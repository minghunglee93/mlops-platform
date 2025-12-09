.PHONY: help install test docker-build k8s-deploy helm-deploy clean

NAMESPACE ?= mlops
IMAGE_TAG ?= latest
RELEASE_NAME ?= mlops-platform

help:
	@echo "MLOps Platform - Available Commands"
	@echo "===================================="
	@echo "Local Development:"
	@echo "  make install          - Install dependencies"
	@echo "  make test            - Run tests"
	@echo "  make train           - Train example model"
	@echo "  make serve           - Start API server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run in Docker"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-deploy      - Deploy to Kubernetes"
	@echo "  make helm-deploy     - Deploy with Helm"
	@echo "  make k8s-status      - Check deployment status"
	@echo "  make k8s-logs        - View API logs"
	@echo "  make k8s-clean       - Delete all resources"
	@echo ""
	@echo "Examples:"
	@echo "  make run-examples    - Run all examples"

install:
	pip install -e .

test:
	pytest tests/ -v --cov=.

train:
	python run_training.py

serve:
	python run_serving.py

docker-build:
	docker build -t mlops-platform:$(IMAGE_TAG) .

docker-run:
	docker run -p 8000:8000 mlops-platform:$(IMAGE_TAG)

k8s-deploy:
	bash scripts/deploy-k8s.sh

helm-deploy:
	bash scripts/deploy-helm.sh

k8s-status:
	kubectl get all -n $(NAMESPACE)

k8s-logs:
	kubectl logs -f deployment/mlops-api -n $(NAMESPACE)

k8s-port-forward:
	kubectl port-forward svc/mlops-api-service 8000:8000 -n $(NAMESPACE)

k8s-clean:
	kubectl delete namespace $(NAMESPACE)

helm-clean:
	helm uninstall $(RELEASE_NAME) -n $(NAMESPACE)

run-examples:
	python run_training.py
	python run_feature_store.py
	python run_drift_detection.py
	python run_ab_testing.py
	python run_retraining.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist build *.egg-info
