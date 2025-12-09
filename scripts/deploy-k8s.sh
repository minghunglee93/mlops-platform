#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================"
echo "MLOps Platform - Kubernetes Deployment"
echo "======================================"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}kubectl not found${NC}"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo -e "${RED}docker not found${NC}"; exit 1; }

# Configuration
NAMESPACE=${NAMESPACE:-mlops}
IMAGE_TAG=${IMAGE_TAG:-latest}
CONTEXT=${CONTEXT:-$(kubectl config current-context)}

echo -e "${GREEN}Using context: ${CONTEXT}${NC}"
echo -e "${GREEN}Namespace: ${NAMESPACE}${NC}"

# Build Docker image
echo -e "\n${YELLOW}Step 1: Building Docker image...${NC}"
docker build -t mlops-platform:${IMAGE_TAG} .
echo -e "${GREEN}✓ Image built${NC}"

# Create namespace
echo -e "\n${YELLOW}Step 2: Creating namespace...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓ Namespace ready${NC}"

# Apply secrets
echo -e "\n${YELLOW}Step 3: Creating secrets...${NC}"
echo -e "${RED}⚠ WARNING: Update secrets before production deployment!${NC}"
kubectl apply -f kubernetes/secrets.yaml -n ${NAMESPACE}
echo -e "${GREEN}✓ Secrets created${NC}"

# Apply ConfigMaps
echo -e "\n${YELLOW}Step 4: Creating ConfigMaps...${NC}"
kubectl apply -f kubernetes/configmap.yaml -n ${NAMESPACE}
echo -e "${GREEN}✓ ConfigMaps created${NC}"

# Apply PVCs
echo -e "\n${YELLOW}Step 5: Creating Persistent Volume Claims...${NC}"
kubectl apply -f kubernetes/pvc.yaml -n ${NAMESPACE}
echo -e "${GREEN}✓ PVCs created${NC}"

# Deploy PostgreSQL
echo -e "\n${YELLOW}Step 6: Deploying PostgreSQL...${NC}"
kubectl apply -f kubernetes/deployment.yaml -n ${NAMESPACE}
echo -e "${GREEN}✓ PostgreSQL deployed${NC}"

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=300s

# Deploy MLflow
echo -e "\n${YELLOW}Step 7: Deploying MLflow...${NC}"
kubectl rollout status statefulset/postgres -n ${NAMESPACE}
echo -e "${GREEN}✓ MLflow deployed${NC}"

# Deploy API
echo -e "\n${YELLOW}Step 8: Deploying API...${NC}"
kubectl rollout status deployment/mlflow -n ${NAMESPACE}
echo -e "${GREEN}✓ API deployed${NC}"

# Deploy Monitoring
echo -e "\n${YELLOW}Step 9: Deploying Monitoring...${NC}"
kubectl rollout status deployment/mlops-api -n ${NAMESPACE}
kubectl apply -f kubernetes/services.yaml -n ${NAMESPACE}
echo -e "${GREEN}✓ Services created${NC}"

# Apply HPA
echo -e "\n${YELLOW}Step 10: Configuring autoscaling...${NC}"
kubectl apply -f kubernetes/hpa.yaml -n ${NAMESPACE}
echo -e "${GREEN}✓ HPA configured${NC}"

# Apply Ingress
echo -e "\n${YELLOW}Step 11: Configuring Ingress...${NC}"
kubectl apply -f kubernetes/ingress.yaml -n ${NAMESPACE}
echo -e "${GREEN}✓ Ingress configured${NC}"

# Check deployment status
echo -e "\n${YELLOW}Checking deployment status...${NC}"
kubectl get pods -n ${NAMESPACE}
kubectl get svc -n ${NAMESPACE}

echo -e "\n${GREEN}======================================"
echo "Deployment Complete!"
echo "======================================${NC}"
echo ""
echo "Access points:"
echo "  API:        http://api.mlops.example.com"
echo "  MLflow:     http://mlflow.mlops.example.com"
echo "  Grafana:    http://grafana.mlops.example.com"
echo ""
echo "Next steps:"
echo "  1. Update DNS records"
echo "  2. Update secrets in production"
echo "  3. Configure cert-manager for TLS"
echo "  4. Review resource limits"
echo ""
echo "Monitoring:"
echo "  kubectl get pods -n ${NAMESPACE}"
echo "  kubectl logs -f deployment/mlops-api -n ${NAMESPACE}"
echo "  kubectl port-forward svc/mlops-api-service 8000:8000 -n ${NAMESPACE}"
