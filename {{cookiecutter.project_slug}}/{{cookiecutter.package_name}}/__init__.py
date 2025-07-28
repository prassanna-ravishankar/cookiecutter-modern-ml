"""{{ cookiecutter.project_name }} - A modern ML project."""

__version__ = "0.1.0"
__author__ = "{{ cookiecutter.author_name }}"
__email__ = "{{ cookiecutter.author_email }}"

from .config import get_settings

__all__ = ["get_settings", "__version__"]