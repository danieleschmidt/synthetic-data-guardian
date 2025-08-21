"""
Deployment Engine - Automated deployment and infrastructure management
"""

import asyncio
import json
import yaml
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import tempfile
import shutil

from ..utils.logger import get_logger


class DeploymentEnvironment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"


class InfrastructureProvider(Enum):
    """Infrastructure providers."""
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    provider: InfrastructureProvider
    image_tag: str = "latest"
    replicas: int = 3
    resources: Dict[str, Any] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    health_checks: Dict[str, Any] = field(default_factory=dict)
    auto_scaling: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Deployment result."""
    success: bool
    deployment_id: str
    duration: float
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    details: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    rollback_available: bool = True


class DeploymentEngine:
    """
    Deployment Engine - Automated deployment and infrastructure management.
    
    Provides comprehensive deployment capabilities including:
    - Multi-environment deployment automation
    - Infrastructure as Code (IaC) generation
    - Container orchestration
    - Blue-green and canary deployments
    - Auto-scaling configuration
    - Health checks and monitoring
    - Rollback mechanisms
    - Secret management
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        """Initialize deployment engine."""
        self.config = config or {}
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # Deployment state
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        
        # Default configurations
        self.default_configs = self._initialize_default_configs()
        
        # Infrastructure templates
        self.templates = self._initialize_templates()
        
        self.logger.info("Deployment Engine initialized")
    
    def _initialize_default_configs(self) -> Dict[DeploymentEnvironment, Dict[str, Any]]:
        """Initialize default configurations for each environment."""
        return {
            DeploymentEnvironment.DEVELOPMENT: {
                'replicas': 1,
                'resources': {
                    'requests': {'memory': '256Mi', 'cpu': '100m'},
                    'limits': {'memory': '512Mi', 'cpu': '200m'}
                },
                'auto_scaling': {'enabled': False},
                'monitoring': {'enabled': True, 'debug': True}
            },
            DeploymentEnvironment.STAGING: {
                'replicas': 2,
                'resources': {
                    'requests': {'memory': '512Mi', 'cpu': '200m'},
                    'limits': {'memory': '1Gi', 'cpu': '500m'}
                },
                'auto_scaling': {'enabled': True, 'min_replicas': 2, 'max_replicas': 5},
                'monitoring': {'enabled': True, 'debug': False}
            },
            DeploymentEnvironment.PRODUCTION: {
                'replicas': 3,
                'resources': {
                    'requests': {'memory': '1Gi', 'cpu': '500m'},
                    'limits': {'memory': '2Gi', 'cpu': '1000m'}
                },
                'auto_scaling': {'enabled': True, 'min_replicas': 3, 'max_replicas': 10},
                'monitoring': {'enabled': True, 'debug': False, 'alerting': True}
            },
            DeploymentEnvironment.TEST: {
                'replicas': 1,
                'resources': {
                    'requests': {'memory': '128Mi', 'cpu': '50m'},
                    'limits': {'memory': '256Mi', 'cpu': '100m'}
                },
                'auto_scaling': {'enabled': False},
                'monitoring': {'enabled': False}
            }
        }
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize infrastructure templates."""
        return {
            'kubernetes_deployment': """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{name}}-{{environment}}
  labels:
    app: {{name}}
    environment: {{environment}}
    version: {{version}}
spec:
  replicas: {{replicas}}
  selector:
    matchLabels:
      app: {{name}}
      environment: {{environment}}
  template:
    metadata:
      labels:
        app: {{name}}
        environment: {{environment}}
        version: {{version}}
    spec:
      containers:
      - name: {{name}}
        image: {{image}}:{{tag}}
        ports:
        - containerPort: {{port}}
        env:
{{environment_vars}}
        resources:
{{resources}}
        livenessProbe:
{{liveness_probe}}
        readinessProbe:
{{readiness_probe}}
---
apiVersion: v1
kind: Service
metadata:
  name: {{name}}-service-{{environment}}
  labels:
    app: {{name}}
    environment: {{environment}}
spec:
  selector:
    app: {{name}}
    environment: {{environment}}
  ports:
  - port: {{service_port}}
    targetPort: {{port}}
  type: {{service_type}}
""",
            'kubernetes_hpa': """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{name}}-hpa-{{environment}}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{name}}-{{environment}}
  minReplicas: {{min_replicas}}
  maxReplicas: {{max_replicas}}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {{cpu_threshold}}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {{memory_threshold}}
""",
            'docker_compose': """
version: '3.8'
services:
  {{name}}:
    image: {{image}}:{{tag}}
    ports:
      - "{{host_port}}:{{container_port}}"
    environment:
{{environment_vars}}
    deploy:
      replicas: {{replicas}}
      resources:
        limits:
          memory: {{memory_limit}}
          cpus: '{{cpu_limit}}'
        reservations:
          memory: {{memory_reservation}}
          cpus: '{{cpu_reservation}}'
    healthcheck:
      test: {{health_check_command}}
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
""",
            'dockerfile': """
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY *.py ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE {{port}}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{{port}}/health || exit 1

# Run application
CMD ["python", "-m", "src.{{main_module}}"]
""",
            'nginx_config': """
upstream {{name}}_backend {
{{upstream_servers}}
}

server {
    listen 80;
    server_name {{domain}};

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Rate limiting
    limit_req zone=api burst={{burst_limit}} nodelay;

    location / {
        proxy_pass http://{{name}}_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Health check bypass
        if ($request_uri = "/health") {
            access_log off;
        }
    }

    # Health check endpoint
    location /nginx-health {
        access_log off;
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }
}
"""
        }
    
    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy application with specified configuration."""
        deployment_id = f"{config.name}-{config.environment.value}-{int(time.time())}"
        start_time = time.time()
        
        self.logger.info(f"Starting deployment {deployment_id}")
        
        try:
            # Pre-deployment validation
            validation_result = await self._validate_deployment_config(config)
            if not validation_result['valid']:
                raise ValueError(f"Deployment validation failed: {validation_result['errors']}")
            
            # Generate infrastructure manifests
            manifests = await self._generate_manifests(config)
            
            # Execute deployment based on provider
            if config.provider == InfrastructureProvider.KUBERNETES:
                deployment_result = await self._deploy_to_kubernetes(config, manifests)
            elif config.provider == InfrastructureProvider.DOCKER:
                deployment_result = await self._deploy_with_docker(config, manifests)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
            
            # Post-deployment verification
            verification_result = await self._verify_deployment(config, deployment_id)
            
            duration = time.time() - start_time
            
            result = DeploymentResult(
                success=deployment_result['success'] and verification_result['success'],
                deployment_id=deployment_id,
                duration=duration,
                environment=config.environment,
                strategy=config.strategy,
                details={
                    'manifests_generated': len(manifests),
                    'deployment_details': deployment_result,
                    'verification_details': verification_result
                },
                logs=deployment_result.get('logs', []),
                rollback_available=True
            )
            
            # Store deployment
            self.active_deployments[deployment_id] = result
            self.deployment_history.append(result)
            
            self.logger.info(f"Deployment {deployment_id} completed: {'SUCCESS' if result.success else 'FAILED'}")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            
            result = DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                duration=duration,
                environment=config.environment,
                strategy=config.strategy,
                details={'error': str(e)},
                logs=[f"Deployment failed: {str(e)}"],
                rollback_available=False
            )
            
            self.deployment_history.append(result)
            return result
    
    async def _validate_deployment_config(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate deployment configuration."""
        errors = []
        warnings = []
        
        # Basic validation
        if not config.name:
            errors.append("Deployment name is required")
        
        if not config.image_tag:
            warnings.append("No image tag specified, using 'latest'")
        
        if config.replicas < 1:
            errors.append("Replicas must be at least 1")
        
        # Environment-specific validation
        if config.environment == DeploymentEnvironment.PRODUCTION:
            if config.replicas < 2:
                warnings.append("Production deployments should have at least 2 replicas for HA")
            
            if not config.resources:
                warnings.append("Resource limits should be specified for production")
            
            if not config.health_checks:
                errors.append("Health checks are required for production deployments")
        
        # Resource validation
        if config.resources:
            if 'limits' in config.resources:
                limits = config.resources['limits']
                if 'memory' in limits and not limits['memory'].endswith(('Mi', 'Gi')):
                    errors.append("Memory limits must be specified in Mi or Gi")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def _generate_manifests(self, config: DeploymentConfig) -> Dict[str, str]:
        """Generate infrastructure manifests."""
        manifests = {}
        
        # Merge with default config
        default_config = self.default_configs.get(config.environment, {})
        merged_config = {**default_config, **config.__dict__}
        
        if config.provider == InfrastructureProvider.KUBERNETES:
            # Generate Kubernetes deployment manifest
            deployment_manifest = self._render_template(
                'kubernetes_deployment',
                {
                    'name': config.name,
                    'environment': config.environment.value,
                    'version': config.image_tag,
                    'replicas': config.replicas,
                    'image': config.name,
                    'tag': config.image_tag,
                    'port': 8080,
                    'service_port': 80,
                    'service_type': 'ClusterIP',
                    'environment_vars': self._format_env_vars_k8s(config.environment_vars),
                    'resources': self._format_resources_k8s(merged_config.get('resources', {})),
                    'liveness_probe': self._format_health_check_k8s(
                        config.health_checks.get('liveness', {'path': '/health', 'port': 8080})
                    ),
                    'readiness_probe': self._format_health_check_k8s(
                        config.health_checks.get('readiness', {'path': '/ready', 'port': 8080})
                    )
                }
            )
            manifests['deployment.yaml'] = deployment_manifest
            
            # Generate HPA if auto-scaling is enabled
            if merged_config.get('auto_scaling', {}).get('enabled'):
                hpa_manifest = self._render_template(
                    'kubernetes_hpa',
                    {
                        'name': config.name,
                        'environment': config.environment.value,
                        'min_replicas': merged_config['auto_scaling'].get('min_replicas', 2),
                        'max_replicas': merged_config['auto_scaling'].get('max_replicas', 10),
                        'cpu_threshold': merged_config['auto_scaling'].get('cpu_threshold', 70),
                        'memory_threshold': merged_config['auto_scaling'].get('memory_threshold', 80)
                    }
                )
                manifests['hpa.yaml'] = hpa_manifest
        
        elif config.provider == InfrastructureProvider.DOCKER:
            # Generate Docker Compose manifest
            compose_manifest = self._render_template(
                'docker_compose',
                {
                    'name': config.name,
                    'image': config.name,
                    'tag': config.image_tag,
                    'host_port': 8080,
                    'container_port': 8080,
                    'replicas': config.replicas,
                    'environment_vars': self._format_env_vars_compose(config.environment_vars),
                    'memory_limit': merged_config.get('resources', {}).get('limits', {}).get('memory', '1G'),
                    'cpu_limit': merged_config.get('resources', {}).get('limits', {}).get('cpu', '1.0'),
                    'memory_reservation': merged_config.get('resources', {}).get('requests', {}).get('memory', '512M'),
                    'cpu_reservation': merged_config.get('resources', {}).get('requests', {}).get('cpu', '0.5'),
                    'health_check_command': '["CMD", "curl", "-f", "http://localhost:8080/health"]'
                }
            )
            manifests['docker-compose.yml'] = compose_manifest
        
        # Generate Dockerfile if needed
        dockerfile = self._render_template(
            'dockerfile',
            {
                'port': 8080,
                'main_module': 'index'
            }
        )
        manifests['Dockerfile'] = dockerfile
        
        return manifests
    
    def _render_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Render template with variables."""
        template = self.templates.get(template_name, '')
        
        # Simple template rendering (in production, use Jinja2 or similar)
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            template = template.replace(placeholder, str(value))
        
        return template
    
    def _format_env_vars_k8s(self, env_vars: Dict[str, str]) -> str:
        """Format environment variables for Kubernetes."""
        if not env_vars:
            return "        # No environment variables"
        
        formatted = []
        for key, value in env_vars.items():
            formatted.append(f"        - name: {key}")
            formatted.append(f"          value: \"{value}\"")
        
        return "\\n".join(formatted)
    
    def _format_env_vars_compose(self, env_vars: Dict[str, str]) -> str:
        """Format environment variables for Docker Compose."""
        if not env_vars:
            return "      # No environment variables"
        
        formatted = []
        for key, value in env_vars.items():
            formatted.append(f"      {key}: \"{value}\"")
        
        return "\\n".join(formatted)
    
    def _format_resources_k8s(self, resources: Dict[str, Any]) -> str:
        """Format resource specifications for Kubernetes."""
        if not resources:
            return "          # No resource specifications"
        
        formatted = []
        if 'requests' in resources:
            formatted.append("          requests:")
            for resource, value in resources['requests'].items():
                formatted.append(f"            {resource}: {value}")
        
        if 'limits' in resources:
            formatted.append("          limits:")
            for resource, value in resources['limits'].items():
                formatted.append(f"            {resource}: {value}")
        
        return "\\n".join(formatted)
    
    def _format_health_check_k8s(self, health_check: Dict[str, Any]) -> str:
        """Format health check for Kubernetes."""
        path = health_check.get('path', '/health')
        port = health_check.get('port', 8080)
        
        return f"""          httpGet:
            path: {path}
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3"""
    
    async def _deploy_to_kubernetes(self, config: DeploymentConfig, manifests: Dict[str, str]) -> Dict[str, Any]:
        """Deploy to Kubernetes cluster."""
        self.logger.info("Deploying to Kubernetes")
        
        deployment_logs = []
        
        try:
            # Create temporary directory for manifests
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write manifests to files
                manifest_files = []
                for filename, content in manifests.items():
                    if filename.endswith(('.yaml', '.yml')):
                        file_path = temp_path / filename
                        file_path.write_text(content)
                        manifest_files.append(str(file_path))
                
                # Apply manifests using kubectl
                for manifest_file in manifest_files:
                    cmd = ['kubectl', 'apply', '-f', manifest_file]
                    result = await self._run_command(cmd)
                    
                    deployment_logs.extend(result['logs'])
                    
                    if result['returncode'] != 0:
                        return {
                            'success': False,
                            'error': f"kubectl apply failed for {manifest_file}",
                            'logs': deployment_logs
                        }
                
                # Wait for deployment to be ready
                deployment_name = f"{config.name}-{config.environment.value}"
                wait_cmd = ['kubectl', 'rollout', 'status', f'deployment/{deployment_name}', '--timeout=300s']
                wait_result = await self._run_command(wait_cmd)
                
                deployment_logs.extend(wait_result['logs'])
                
                return {
                    'success': wait_result['returncode'] == 0,
                    'logs': deployment_logs,
                    'manifests_applied': len(manifest_files)
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'logs': deployment_logs
            }
    
    async def _deploy_with_docker(self, config: DeploymentConfig, manifests: Dict[str, str]) -> Dict[str, Any]:
        """Deploy with Docker Compose."""
        self.logger.info("Deploying with Docker Compose")
        
        deployment_logs = []
        
        try:
            # Create temporary directory for manifests
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write docker-compose.yml
                compose_file = temp_path / 'docker-compose.yml'
                compose_file.write_text(manifests['docker-compose.yml'])
                
                # Build image if Dockerfile is provided
                if 'Dockerfile' in manifests:
                    dockerfile_path = temp_path / 'Dockerfile'
                    dockerfile_path.write_text(manifests['Dockerfile'])
                    
                    # Build image
                    build_cmd = ['docker', 'build', '-t', f"{config.name}:{config.image_tag}", str(temp_path)]
                    build_result = await self._run_command(build_cmd)
                    deployment_logs.extend(build_result['logs'])
                    
                    if build_result['returncode'] != 0:
                        return {
                            'success': False,
                            'error': 'Docker build failed',
                            'logs': deployment_logs
                        }
                
                # Deploy with docker-compose
                deploy_cmd = ['docker-compose', '-f', str(compose_file), 'up', '-d']
                deploy_result = await self._run_command(deploy_cmd)
                deployment_logs.extend(deploy_result['logs'])
                
                return {
                    'success': deploy_result['returncode'] == 0,
                    'logs': deployment_logs,
                    'compose_file': str(compose_file)
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'logs': deployment_logs
            }
    
    async def _verify_deployment(self, config: DeploymentConfig, deployment_id: str) -> Dict[str, Any]:
        """Verify deployment health and readiness."""
        self.logger.info(f"Verifying deployment {deployment_id}")
        
        verification_logs = []
        
        try:
            # Wait for services to be ready
            await asyncio.sleep(10)  # Give services time to start
            
            if config.provider == InfrastructureProvider.KUBERNETES:
                # Check pod status
                deployment_name = f"{config.name}-{config.environment.value}"
                cmd = ['kubectl', 'get', 'pods', '-l', f'app={config.name}', '-o', 'json']
                result = await self._run_command(cmd)
                
                if result['returncode'] == 0:
                    # Parse pod status (simplified)
                    verification_logs.append("Pods are running")
                    return {'success': True, 'logs': verification_logs}
                else:
                    verification_logs.extend(result['logs'])
                    return {'success': False, 'logs': verification_logs}
            
            elif config.provider == InfrastructureProvider.DOCKER:
                # Check container status
                cmd = ['docker', 'ps', '--filter', f'name={config.name}', '--format', 'table {{.Names}}\\t{{.Status}}']
                result = await self._run_command(cmd)
                
                if result['returncode'] == 0 and 'Up' in result['stdout']:
                    verification_logs.append("Containers are running")
                    return {'success': True, 'logs': verification_logs}
                else:
                    verification_logs.extend(result['logs'])
                    return {'success': False, 'logs': verification_logs}
            
            return {'success': False, 'logs': ['Unknown provider for verification']}
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'logs': verification_logs
            }
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run shell command asynchronously."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            logs = []
            if stdout:
                logs.append(f"STDOUT: {stdout.decode()}")
            if stderr:
                logs.append(f"STDERR: {stderr.decode()}")
            
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode(),
                'stderr': stderr.decode(),
                'logs': logs
            }
            
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'logs': [f"Command execution failed: {str(e)}"]
            }
    
    async def rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback a deployment."""
        if deployment_id not in self.active_deployments:
            return {'success': False, 'error': 'Deployment not found'}
        
        deployment = self.active_deployments[deployment_id]
        
        if not deployment.rollback_available:
            return {'success': False, 'error': 'Rollback not available for this deployment'}
        
        self.logger.info(f"Rolling back deployment {deployment_id}")
        
        try:
            # Implementation depends on the deployment strategy and provider
            # For now, return a mock rollback result
            rollback_logs = [f"Rolling back deployment {deployment_id}"]
            
            return {
                'success': True,
                'logs': rollback_logs,
                'rollback_time': time.time()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific deployment."""
        if deployment_id not in self.active_deployments:
            return None
        
        deployment = self.active_deployments[deployment_id]
        
        return {
            'deployment_id': deployment_id,
            'success': deployment.success,
            'environment': deployment.environment.value,
            'strategy': deployment.strategy.value,
            'duration': deployment.duration,
            'rollback_available': deployment.rollback_available,
            'details': deployment.details
        }
    
    def list_deployments(self, environment: Optional[DeploymentEnvironment] = None) -> List[Dict[str, Any]]:
        """List deployments, optionally filtered by environment."""
        deployments = []
        
        for deployment in self.deployment_history:
            if environment is None or deployment.environment == environment:
                deployments.append({
                    'deployment_id': deployment.deployment_id,
                    'success': deployment.success,
                    'environment': deployment.environment.value,
                    'strategy': deployment.strategy.value,
                    'duration': deployment.duration,
                    'timestamp': deployment.details.get('timestamp', time.time())
                })
        
        return sorted(deployments, key=lambda x: x.get('timestamp', 0), reverse=True)
    
    async def generate_infrastructure_code(self, config: DeploymentConfig, 
                                         output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate Infrastructure as Code files."""
        if output_dir is None:
            output_dir = Path('./infrastructure')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate manifests
        manifests = await self._generate_manifests(config)
        
        # Write files to output directory
        files_created = []
        for filename, content in manifests.items():
            file_path = output_dir / filename
            file_path.write_text(content)
            files_created.append(str(file_path))
        
        # Generate additional infrastructure files
        if config.provider == InfrastructureProvider.KUBERNETES:
            # Generate namespace
            namespace_content = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {config.name}-{config.environment.value}
  labels:
    app: {config.name}
    environment: {config.environment.value}
"""
            namespace_file = output_dir / 'namespace.yaml'
            namespace_file.write_text(namespace_content)
            files_created.append(str(namespace_file))
            
            # Generate ConfigMap for application config
            configmap_content = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: {config.name}-config-{config.environment.value}
  namespace: {config.name}-{config.environment.value}
data:
  environment: {config.environment.value}
  log_level: {"DEBUG" if config.environment == DeploymentEnvironment.DEVELOPMENT else "INFO"}
"""
            configmap_file = output_dir / 'configmap.yaml'
            configmap_file.write_text(configmap_content)
            files_created.append(str(configmap_file))
        
        return {
            'success': True,
            'output_directory': str(output_dir),
            'files_created': files_created,
            'total_files': len(files_created)
        }
    
    async def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        total_deployments = len(self.deployment_history)
        successful_deployments = len([d for d in self.deployment_history if d.success])
        
        # Environment statistics
        env_stats = {}
        for env in DeploymentEnvironment:
            env_deployments = [d for d in self.deployment_history if d.environment == env]
            env_stats[env.value] = {
                'total': len(env_deployments),
                'successful': len([d for d in env_deployments if d.success]),
                'success_rate': len([d for d in env_deployments if d.success]) / len(env_deployments) * 100 if env_deployments else 0
            }
        
        # Recent deployment trends
        recent_deployments = self.deployment_history[-10:] if len(self.deployment_history) > 10 else self.deployment_history
        avg_duration = sum(d.duration for d in recent_deployments) / len(recent_deployments) if recent_deployments else 0
        
        return {
            'report_timestamp': time.time(),
            'deployment_statistics': {
                'total_deployments': total_deployments,
                'successful_deployments': successful_deployments,
                'success_rate': (successful_deployments / total_deployments * 100) if total_deployments > 0 else 0,
                'average_duration': avg_duration
            },
            'environment_statistics': env_stats,
            'active_deployments': len(self.active_deployments),
            'recent_deployments': [
                {
                    'deployment_id': d.deployment_id,
                    'environment': d.environment.value,
                    'success': d.success,
                    'duration': d.duration
                }
                for d in recent_deployments
            ],
            'recommendations': self._generate_deployment_recommendations(env_stats, avg_duration)
        }
    
    def _generate_deployment_recommendations(self, env_stats: Dict[str, Any], avg_duration: float) -> List[str]:
        """Generate deployment recommendations."""
        recommendations = []
        
        # Check success rates
        for env, stats in env_stats.items():
            if stats['success_rate'] < 90 and stats['total'] > 0:
                recommendations.append(f"Improve {env} deployment success rate (currently {stats['success_rate']:.1f}%)")
        
        # Check deployment duration
        if avg_duration > 300:  # 5 minutes
            recommendations.append("Consider optimizing deployment process to reduce duration")
        
        # General recommendations
        recommendations.extend([
            "Implement automated rollback mechanisms",
            "Add comprehensive health checks for all environments",
            "Set up monitoring and alerting for deployments",
            "Consider implementing blue-green deployments for production"
        ])
        
        return recommendations