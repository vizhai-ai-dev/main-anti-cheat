import os
import json
import yaml
from typing import Dict, Any, List, Optional
from datetime import datetime
import inspect
import importlib
import pydoc
import markdown
import shutil

class DocumentationGenerator:
    """Class for generating project documentation"""
    
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.docs_dir = os.path.join(project_dir, 'docs')
        self.api_dir = os.path.join(self.docs_dir, 'api')
        self.examples_dir = os.path.join(self.docs_dir, 'examples')
        
        # Create documentation directories
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.api_dir, exist_ok=True)
        os.makedirs(self.examples_dir, exist_ok=True)
    
    def generate_module_docs(self, module_name: str) -> str:
        """Generate documentation for a module"""
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return f"Error: Could not import module {module_name}"
        
        doc = f"# {module_name}\n\n"
        
        # Module docstring
        if module.__doc__:
            doc += f"{module.__doc__}\n\n"
        
        # Classes
        classes = inspect.getmembers(module, inspect.isclass)
        if classes:
            doc += "## Classes\n\n"
            for name, cls in classes:
                doc += f"### {name}\n\n"
                if cls.__doc__:
                    doc += f"{cls.__doc__}\n\n"
                
                # Methods
                methods = inspect.getmembers(cls, inspect.isfunction)
                if methods:
                    doc += "#### Methods\n\n"
                    for method_name, method in methods:
                        doc += f"##### {method_name}\n\n"
                        if method.__doc__:
                            doc += f"{method.__doc__}\n\n"
                        
                        # Parameters
                        sig = inspect.signature(method)
                        if sig.parameters:
                            doc += "Parameters:\n\n"
                            for param_name, param in sig.parameters.items():
                                doc += f"- {param_name}: {param.annotation}\n"
                            doc += "\n"
        
        # Functions
        functions = inspect.getmembers(module, inspect.isfunction)
        if functions:
            doc += "## Functions\n\n"
            for name, func in functions:
                doc += f"### {name}\n\n"
                if func.__doc__:
                    doc += f"{func.__doc__}\n\n"
                
                # Parameters
                sig = inspect.signature(func)
                if sig.parameters:
                    doc += "Parameters:\n\n"
                    for param_name, param in sig.parameters.items():
                        doc += f"- {param_name}: {param.annotation}\n"
                    doc += "\n"
        
        return doc
    
    def generate_api_docs(self) -> None:
        """Generate API documentation"""
        modules = [
            'screen_switch',
            'gaze_tracking',
            'multi_person',
            'audio_analysis',
            'cheat_score',
            'run_all'
        ]
        
        for module_name in modules:
            doc = self.generate_module_docs(module_name)
            output_path = os.path.join(self.api_dir, f"{module_name}.md")
            
            with open(output_path, 'w') as f:
                f.write(doc)
    
    def generate_example_docs(self) -> None:
        """Generate example documentation"""
        examples = [
            {
                'name': 'Basic Usage',
                'description': 'Basic usage of the anti-model system',
                'code': '''
from run_all import run_all_modules

# Run all detection modules
result = run_all_modules('video.mp4')
print(result)
'''
            },
            {
                'name': 'Custom Configuration',
                'description': 'Using custom configuration for detection',
                'code': '''
from screen_switch import detect_screen_switch
from gaze_tracking import track_gaze

# Run individual modules with custom settings
screen_result = detect_screen_switch(
    'video.mp4',
    threshold=0.7,
    sample_interval=0.5
)

gaze_result = track_gaze(
    'video.mp4',
    confidence_threshold=0.6
)
'''
            }
        ]
        
        for example in examples:
            doc = f"# {example['name']}\n\n"
            doc += f"{example['description']}\n\n"
            doc += "```python\n"
            doc += example['code'].strip()
            doc += "\n```\n"
            
            output_path = os.path.join(
                self.examples_dir,
                f"{example['name'].lower().replace(' ', '_')}.md"
            )
            
            with open(output_path, 'w') as f:
                f.write(doc)
    
    def generate_readme(self) -> None:
        """Generate main README file"""
        readme = """# Anti-Model

A system for detecting cheating behavior in video recordings.

## Features

- Screen switch detection
- Gaze tracking
- Multi-person detection
- Audio analysis
- Cheat score calculation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from run_all import run_all_modules

# Run all detection modules
result = run_all_modules('video.mp4')
print(result)
```

## Documentation

- [API Documentation](docs/api/)
- [Examples](docs/examples/)

## License

MIT License
"""
        
        output_path = os.path.join(self.project_dir, 'README.md')
        with open(output_path, 'w') as f:
            f.write(readme)
    
    def generate_docs(self) -> None:
        """Generate all documentation"""
        self.generate_api_docs()
        self.generate_example_docs()
        self.generate_readme()
        
        # Generate HTML documentation
        self.generate_html_docs()
    
    def generate_html_docs(self) -> None:
        """Generate HTML documentation"""
        html_dir = os.path.join(self.docs_dir, 'html')
        os.makedirs(html_dir, exist_ok=True)
        
        # Convert markdown to HTML
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith('.md'):
                    md_path = os.path.join(root, file)
                    html_path = os.path.join(
                        html_dir,
                        os.path.relpath(md_path, self.docs_dir).replace('.md', '.html')
                    )
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(html_path), exist_ok=True)
                    
                    # Convert markdown to HTML
                    with open(md_path, 'r') as f:
                        md_content = f.read()
                    
                    html_content = markdown.markdown(
                        md_content,
                        extensions=['extra', 'codehilite']
                    )
                    
                    # Add HTML template
                    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Anti-Model Documentation</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/4.0.0/github-markdown.min.css">
    <style>
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
        }}
    </style>
</head>
<body>
    <article class="markdown-body">
        {html_content}
    </article>
</body>
</html>"""
                    
                    with open(html_path, 'w') as f:
                        f.write(html_template)

def generate_docs(project_dir: str) -> None:
    """Generate project documentation"""
    generator = DocumentationGenerator(project_dir)
    generator.generate_docs()

if __name__ == '__main__':
    generate_docs(os.getcwd()) 