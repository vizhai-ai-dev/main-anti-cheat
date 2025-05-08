import os
import shutil
import subprocess
from typing import List, Dict, Any, Optional
import json
import yaml
from datetime import datetime
import logging
import tarfile
import zipfile

class DeploymentManager:
    """Class for managing project deployment"""
    
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.dist_dir = os.path.join(project_dir, 'dist')
        self.build_dir = os.path.join(project_dir, 'build')
        
        # Create distribution directories
        os.makedirs(self.dist_dir, exist_ok=True)
        os.makedirs(self.build_dir, exist_ok=True)
    
    def clean_build(self) -> None:
        """Clean build directories"""
        if os.path.exists(self.build_dir):
            shutil.rmtree(self.build_dir)
        if os.path.exists(self.dist_dir):
            shutil.rmtree(self.dist_dir)
        
        os.makedirs(self.build_dir)
        os.makedirs(self.dist_dir)
    
    def build_package(self) -> None:
        """Build Python package"""
        try:
            subprocess.run(
                ['python', 'setup.py', 'sdist', 'bdist_wheel'],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error building package: {str(e)}")
    
    def create_docker_image(self, tag: str = 'latest') -> None:
        """Create Docker image"""
        dockerfile = """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "run_all:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        # Write Dockerfile
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile)
        
        try:
            subprocess.run(
                ['docker', 'build', '-t', f'anti-model:{tag}', '.'],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error building Docker image: {str(e)}")
    
    def create_release_archive(self, version: str) -> str:
        """Create release archive"""
        archive_name = f'anti-model-{version}'
        archive_path = os.path.join(self.dist_dir, archive_name)
        
        # Create archive
        with tarfile.open(f'{archive_path}.tar.gz', 'w:gz') as tar:
            tar.add(self.project_dir, arcname=archive_name)
        
        return f'{archive_path}.tar.gz'
    
    def create_windows_installer(self, version: str) -> str:
        """Create Windows installer"""
        installer_name = f'anti-model-{version}-setup'
        installer_path = os.path.join(self.dist_dir, installer_name)
        
        # Create NSIS script
        nsis_script = f"""
!include "MUI2.nsh"

Name "Anti-Model {version}"
OutFile "{installer_path}.exe"
InstallDir "$PROGRAMFILES\\Anti-Model"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath "$INSTDIR"
    File /r "{self.project_dir}\\*.*"
    
    WriteUninstaller "$INSTDIR\\uninstall.exe"
    
    CreateDirectory "$SMPROGRAMS\\Anti-Model"
    CreateShortcut "$SMPROGRAMS\\Anti-Model\\Uninstall.lnk" "$INSTDIR\\uninstall.exe"
SectionEnd

Section "Uninstall"
    RMDir /r "$INSTDIR"
    RMDir /r "$SMPROGRAMS\\Anti-Model"
SectionEnd
"""
        
        # Write NSIS script
        with open('installer.nsi', 'w') as f:
            f.write(nsis_script)
        
        try:
            subprocess.run(['makensis', 'installer.nsi'], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error creating Windows installer: {str(e)}")
        
        return f'{installer_path}.exe'
    
    def deploy_to_pypi(self, username: str, password: str) -> None:
        """Deploy package to PyPI"""
        try:
            subprocess.run(
                [
                    'twine',
                    'upload',
                    '--username', username,
                    '--password', password,
                    'dist/*'
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error deploying to PyPI: {str(e)}")
    
    def create_deployment_config(self, config: Dict[str, Any]) -> None:
        """Create deployment configuration file"""
        config_path = os.path.join(self.project_dir, 'deployment.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def create_kubernetes_config(self) -> None:
        """Create Kubernetes configuration"""
        k8s_config = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'anti-model',
                'labels': {
                    'app': 'anti-model'
                }
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'matchLabels': {
                        'app': 'anti-model'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'anti-model'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'anti-model',
                            'image': 'anti-model:latest',
                            'ports': [{
                                'containerPort': 8000
                            }]
                        }]
                    }
                }
            }
        }
        
        # Write Kubernetes config
        k8s_path = os.path.join(self.project_dir, 'kubernetes.yaml')
        with open(k8s_path, 'w') as f:
            yaml.dump(k8s_config, f, default_flow_style=False)
    
    def create_github_workflow(self) -> None:
        """Create GitHub Actions workflow"""
        workflow = """
name: CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m unittest discover tests

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build package
      run: |
        python setup.py sdist bdist_wheel
    - name: Upload artifacts
      uses: actions/upload-artifact@v2
      with:
        name: dist
        path: dist/

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v2
    - name: Download artifacts
      uses: actions/download-artifact@v2
      with:
        name: dist
        path: dist
    - name: Deploy to PyPI
      env:
        TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
        TWINE_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
      run: |
        pip install twine
        twine upload dist/*
"""
        
        # Create workflows directory
        workflows_dir = os.path.join(self.project_dir, '.github', 'workflows')
        os.makedirs(workflows_dir, exist_ok=True)
        
        # Write workflow file
        workflow_path = os.path.join(workflows_dir, 'ci-cd.yaml')
        with open(workflow_path, 'w') as f:
            f.write(workflow)

def create_deployment(project_dir: str) -> None:
    """Create deployment configuration"""
    manager = DeploymentManager(project_dir)
    
    # Clean build directories
    manager.clean_build()
    
    # Build package
    manager.build_package()
    
    # Create Docker image
    manager.create_docker_image()
    
    # Create Kubernetes config
    manager.create_kubernetes_config()
    
    # Create GitHub workflow
    manager.create_github_workflow()
    
    # Create deployment config
    config = {
        'version': '1.0.0',
        'docker': {
            'image': 'anti-model',
            'tag': 'latest'
        },
        'kubernetes': {
            'replicas': 3,
            'resources': {
                'requests': {
                    'cpu': '500m',
                    'memory': '512Mi'
                },
                'limits': {
                    'cpu': '1000m',
                    'memory': '1Gi'
                }
            }
        }
    }
    manager.create_deployment_config(config)

if __name__ == '__main__':
    create_deployment(os.getcwd()) 