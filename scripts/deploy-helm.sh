#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "MLOps Platform - Helm Deployment"
echo "=================================="

# Check prerequisites
command -v helm >/dev/null 2>&1 || { echo -e "${RED}helm not found${NC}"; exit 1; }

NAMESPACE=${NAMESPACE:-mlops}
RELEASE_NAME=${RELEASE_NAME:-mlops-platform}
VALUES_FILE=${VALUES_FILE:-helm/mlops-platform/values.yaml}

echo -e "${GREEN}Release: ${RELEASE_NAME}${NC}"
echo -e "${GREEN}Namespace: ${NAMESPACE}${NC}"

# Create namespace
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Add dependencies
echo -e "\n${YELLOW}Adding Helm repositories...${NC}"
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install/Upgrade
echo -e "\n${YELLOW}Deploying with Helm...${NC}"
helm upgrade --install ${RELEASE_NAME} ./helm/mlops-platform \
  --namespace ${NAMESPACE} \
  --values ${VALUES_FILE} \
  --create-namespace \
  --wait \
  --timeout 10m

# Status
echo -e "\n${GREEN}Deployment complete!${NC}"
helm status ${RELEASE_NAME} -n ${NAMESPACE}

echo -e "\n${YELLOW}Resources:${NC}"
kubectl get all -n ${NAMESPACE}
