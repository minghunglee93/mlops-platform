# Kubernetes Deployment Guide

## Quick Start

```bash
# Deploy with kubectl
bash scripts/deploy-k8s.sh

# Or deploy with Helm
bash scripts/deploy-helm.sh
```

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Docker
- Helm 3 (for Helm deployment)
- At least 8GB RAM, 4 CPUs available

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Ingress                           │
│  (api.mlops.com, mlflow.mlops.com, grafana.mlops.com)│
└────────────┬────────────────────────────────────────┘
             │
     ┌───────┼────────┬────────────┐
     ▼       ▼        ▼            ▼
  ┌─────┐ ┌─────┐ ┌────────┐  ┌─────────┐
  │ API │ │MLflow│ │Prometheus│ │ Grafana │
  │ Pods│ │ Pod │ │  Pod    │  │  Pod    │
  └──┬──┘ └──┬──┘ └────────┘  └─────────┘
     │       │
     └───┬───┘
         ▼
    ┌──────────┐
    │PostgreSQL│
    └──────────┘
```

## Deployment Options

### Option 1: kubectl (Raw Manifests)

```bash
# Set environment
export NAMESPACE=mlops
export IMAGE_TAG=latest

# Deploy all components
bash scripts/deploy-k8s.sh

# Or manually
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/pvc.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/services.yaml
kubectl apply -f kubernetes/hpa.yaml
kubectl apply -f kubernetes/ingress.yaml
```

### Option 2: Helm Chart

```bash
# Install
helm install mlops-platform ./helm/mlops-platform \
  --namespace mlops \
  --create-namespace \
  --values helm/mlops-platform/values.yaml

# Upgrade
helm upgrade mlops-platform ./helm/mlops-platform \
  --namespace mlops \
  --values helm/mlops-platform/values.yaml

# Uninstall
helm uninstall mlops-platform -n mlops
```

## Configuration

### Secrets Management

**Development:**
```bash
kubectl create secret generic mlops-secrets \
  --from-literal=postgres-password=yourpass \
  --from-literal=grafana-password=yourpass \
  -n mlops
```

**Production:** Use External Secrets Operator or Sealed Secrets

```yaml
# external-secrets.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: mlops-secrets
spec:
  secretStoreRef:
    name: aws-secrets-manager
  data:
  - secretKey: postgres-password
    remoteRef:
      key: mlops/postgres-password
```

### Resource Limits

Adjust in `values.yaml`:
```yaml
resources:
  limits:
    cpu: 2000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 512Mi
```

### Storage Classes

Update based on your cloud provider:
```yaml
# AWS EBS
storageClassName: gp3

# GCP Persistent Disk
storageClassName: pd-ssd

# Azure Disk
storageClassName: managed-premium
```

## Horizontal Pod Autoscaling

Configure scaling thresholds:

```yaml
autoscaling:
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

Monitor scaling:
```bash
kubectl get hpa -n mlops -w
```

## Ingress & TLS

### Install NGINX Ingress Controller

```bash
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --create-namespace \
  --namespace ingress-nginx
```

### Install cert-manager

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Configure ClusterIssuer

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

## Monitoring

### Access Dashboards

```bash
# Port-forward Grafana
kubectl port-forward svc/grafana-service 3000:3000 -n mlops

# Port-forward Prometheus
kubectl port-forward svc/prometheus-service 9090:9090 -n mlops

# Access at localhost:3000 and localhost:9090
```

### View Logs

```bash
# API logs
kubectl logs -f deployment/mlops-api -n mlops

# MLflow logs
kubectl logs -f deployment/mlflow -n mlops

# All pods
kubectl logs -f -l app=mlops-api -n mlops
```

### Metrics

```bash
# Pod metrics
kubectl top pods -n mlops

# Node metrics
kubectl top nodes

# HPA status
kubectl get hpa -n mlops
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n mlops

# Describe pod
kubectl describe pod <pod-name> -n mlops

# Check events
kubectl get events -n mlops --sort-by='.lastTimestamp'
```

### Storage Issues

```bash
# Check PVC status
kubectl get pvc -n mlops

# Describe PVC
kubectl describe pvc mlops-models-pvc -n mlops

# Check storage class
kubectl get storageclass
```

### Database Connection Issues

```bash
# Check PostgreSQL logs
kubectl logs -f statefulset/postgres -n mlops

# Test connection
kubectl run -it --rm debug --image=postgres:15 --restart=Never -n mlops -- \
  psql -h postgres-service -U mlflow -d mlflow
```

### Ingress Not Working

```bash
# Check ingress
kubectl get ingress -n mlops
kubectl describe ingress mlops-ingress -n mlops

# Check cert-manager
kubectl get certificate -n mlops
kubectl describe certificate mlops-tls -n mlops
```

## Maintenance

### Rolling Updates

```bash
# Update image
kubectl set image deployment/mlops-api api=mlops-platform:v2 -n mlops

# Check rollout
kubectl rollout status deployment/mlops-api -n mlops

# Rollback if needed
kubectl rollout undo deployment/mlops-api -n mlops
```

### Backups

```bash
# Backup PostgreSQL
kubectl exec -it postgres-0 -n mlops -- \
  pg_dump -U mlflow mlflow > backup-$(date +%Y%m%d).sql

# Backup PVCs (use velero or similar)
velero backup create mlops-backup --include-namespaces mlops
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment/mlops-api --replicas=5 -n mlops

# Update HPA
kubectl edit hpa mlops-api-hpa -n mlops
```

## Production Checklist

- [ ] Update all secrets with strong passwords
- [ ] Configure external secrets management
- [ ] Set up TLS certificates
- [ ] Configure DNS records
- [ ] Enable resource quotas
- [ ] Set up backups
- [ ] Configure monitoring alerts
- [ ] Enable network policies
- [ ] Review security contexts
- [ ] Set up CI/CD integration
- [ ] Test disaster recovery
- [ ] Document runbook

## Cloud-Specific Guides

### AWS EKS

```bash
# Create cluster
eksctl create cluster --name mlops-cluster --region us-west-2

# Deploy
bash scripts/deploy-k8s.sh
```

### GCP GKE

```bash
# Create cluster
gcloud container clusters create mlops-cluster --zone us-central1-a

# Deploy
bash scripts/deploy-k8s.sh
```

### Azure AKS

```bash
# Create cluster
az aks create --name mlops-cluster --resource-group mlops-rg

# Get credentials
az aks get-credentials --name mlops-cluster --resource-group mlops-rg

# Deploy
bash scripts/deploy-k8s.sh
```

## Performance Optimization

### Node Affinity

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: workload
          operator: In
          values:
          - ml
```

### Pod Disruption Budget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: mlops-api-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: mlops-api
```

## Security

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
spec:
  podSelector:
    matchLabels:
      app: mlops-api
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
```

### RBAC

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mlops-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
```

## Next Steps

1. Set up CI/CD pipelines
2. Configure monitoring alerts
3. Implement multi-region deployment
4. Add service mesh (Istio/Linkerd)
5. Enable GitOps with ArgoCD
