"""{{ cookiecutter.project_short_description }}"""

__version__ = "{{ cookiecutter.project_version }}"
__author__ = "{{ cookiecutter.author_name }}"
__email__ = "{{ cookiecutter.author_email }}"

from .config import get_settings

__all__ = ["get_settings", "__version__"]