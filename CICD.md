# CI/CD Pipelines

## Overview

Automated pipelines for testing, building, and deploying the MLOps platform.

## Workflows

### 1. Test Pipeline (`test.yml`)

Runs on every PR and push:
- **Linting**: flake8, black, isort, mypy
- **Unit Tests**: pytest with coverage (Python 3.9, 3.10, 3.11)
- **Integration Tests**: With PostgreSQL
- **Security Scans**: bandit, safety

### 2. Build & Push (`build-and-push.yml`)

Triggers on main branch push or release:
- Build multi-arch Docker images (amd64, arm64)
- Push to GitHub Container Registry
- Generate SBOM (Software Bill of Materials)
- Scan for vulnerabilities (Trivy)

### 3. Deploy (`deploy.yml`)

Manual deployment with validation:
- Model performance validation
- Deploy to dev/staging/production
- Smoke tests post-deployment
- Automatic rollback on failure

### 4. Scheduled Retraining (`scheduled-retrain.yml`)

Weekly automated retraining:
- Check for drift
- Retrain if needed
- Validate new model
- Auto-promote if better

## Setup

### Required Secrets

```bash
# GitHub Settings → Secrets → Actions

KUBECONFIG              # Kubernetes config for deployment
MLFLOW_TRACKING_URI     # MLflow server URL
SLACK_WEBHOOK          # Notifications (optional)
```

### Branch Protection

```yaml
# Settings → Branches → Add rule
Required status checks:
  - lint
  - test (3.9, 3.10, 3.11)
  - integration-test
  - security-scan
```

## Usage

### Run Tests Locally

```bash
# Lint
flake8 .
black --check .
isort --check-only .

# Test
pytest tests/ -v --cov=.

# Security
bandit -r .
safety check
```

### Build & Push

```bash
# Automatic on push to main
git push origin main

# Or manual
docker build -t ghcr.io/org/mlops-platform:v1.0.0 .
docker push ghcr.io/org/mlops-platform:v1.0.0
```

### Deploy

```bash
# Via GitHub UI:
# Actions → Deploy → Run workflow
# Select environment and image tag

# Or via gh CLI:
gh workflow run deploy.yml \
  -f environment=staging \
  -f image_tag=v1.0.0
```

### Trigger Retraining

```bash
# Manual trigger
gh workflow run scheduled-retrain.yml \
  -f model_name=classifier_model
```

## Multi-Environment Strategy

```
┌─────────┐    ┌─────────┐    ┌────────────┐
│   Dev   │───▶│ Staging │───▶│ Production │
└─────────┘    └─────────┘    └────────────┘
   Auto         Manual          Manual
   Tests      Validation      Final Approval
```

### Dev
- Auto-deploy on merge to `develop`
- Basic smoke tests
- Rapid iteration

### Staging
- Manual trigger
- Full model validation
- Performance tests
- Mirror production config

### Production
- Manual approval required
- Stricter validation gates
- Blue-green deployment
- Automatic rollback

## Model Validation Gates

```python
# Production requirements
accuracy >= 0.85
f1_score >= 0.80
precision >= 0.80
recall >= 0.75
```

Configured in `scripts/validate_model.py`.

## Rollback Strategy

Automatic rollback if:
- Smoke tests fail
- Health checks fail
- Error rate > 5%

```bash
# Manual rollback
kubectl rollout undo deployment/mlops-api -n mlops-production
```

## Monitoring Deployments

```bash
# View workflow runs
gh run list --workflow=deploy.yml

# Watch logs
gh run watch

# Check deployment
kubectl rollout status deployment/mlops-api -n mlops-staging
```

## Best Practices

1. **Test Locally First**
   ```bash
   make test
   make docker-build
   ```

2. **Use Feature Branches**
   ```bash
   git checkout -b feature/new-model
   # PR triggers tests automatically
   ```

3. **Semantic Versioning**
   ```bash
   git tag v1.2.3
   git push --tags
   # Triggers release build
   ```

4. **Monitor Deployments**
   - Check Grafana dashboards
   - Review logs in Kubernetes
   - Verify metrics in Prometheus

5. **Gradual Rollout**
   ```bash
   # Deploy to dev → staging → 10% prod → 100% prod
   ```

## Troubleshooting

### Tests Failing

```bash
# View logs
gh run view <run-id> --log-failed

# Reproduce locally
docker run -it mlops-platform:latest pytest tests/
```

### Build Failures

```bash
# Check Dockerfile
docker build --progress=plain .

# Verify dependencies
pip install -r requirements.txt
```

### Deployment Issues

```bash
# Check pod status
kubectl get pods -n mlops-staging
kubectl describe pod <pod-name> -n mlops-staging

# View deployment logs
kubectl logs -f deployment/mlops-api -n mlops-staging
```

## Advanced: GitOps with ArgoCD

```yaml
# argocd-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: mlops-platform
spec:
  source:
    repoURL: https://github.com/org/mlops-platform
    path: kubernetes
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: mlops
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## Metrics

Track CI/CD performance:
- Build time
- Test success rate
- Deployment frequency
- Time to deploy
- Rollback rate

Dashboard: Grafana → CI/CD Metrics
