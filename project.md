
### \#\# 1. Overview & Vision

  * **Product Name:** `modern-ml-cookiecutter`
  * **Vision:** To be the definitive starting point for modern machine learning projects by providing a high-performance, end-to-end template that accelerates development from local prototyping to scalable cloud deployment.
  * **Motto:** Develop locally, scale to the cloud, and deploy with confidence. 🚀

-----

### \#\# 2. Problem Statement

  * **The Problem:** Starting a new ML project is slow and complex. Engineers spend significant time setting up virtual environments, managing dependencies, configuring linters, writing CI/CD pipelines, and creating boilerplate for training and deployment. This setup is often inconsistent across projects and teams, leading to poor reproducibility and a slow "code-to-cloud" lifecycle.
  * **How We Solve It:** This cookiecutter provides a pre-configured, opinionated template that automates the entire setup process. By integrating a best-in-class toolchain for dependency management (**uv**), training (**Accelerate**), cloud orchestration (**SkyPilot**), and serving (**LitServe**), it allows developers to focus entirely on modeling logic rather than infrastructure.

-----

### \#\# 3. Target Audience

  * **Machine Learning Engineers:** Professionals building and deploying production-level ML systems.
  * **Data Scientists & AI Researchers:** Practitioners who need a reproducible and scalable environment for experimentation and fine-tuning models.

-----

### \#\# 4. Core Features & Requirements

#### **FR-1: Project Generation & Tooling**

  * **FR-1.1:** The template **must** be a standard `cookiecutter` project.
  * **FR-1.2:** It **must** prompt the user for essential variables: `project_name`, `author_name`, `model_checkpoint`, and `default_cloud` (gcp, aws, azure).
  * **FR-1.3:** **Dependency Management:** All Python dependencies **must** be managed via **uv** through a `pyproject.toml` file.
  * **FR-1.4:** **Task Runner:** Common project commands (lint, test, train, serve) **must** be managed by **taskipy** and be executable via the integrated `uv run <task-name>` command.
  * **FR-1.5:** **Code Quality:** The project **must** come pre-configured with **ruff** for linting/formatting and **mypy** for static type checking.
  * **FR-1.6:** **CI/CD:** A GitHub Actions workflow **must** be included to automate linting and testing on every push and pull request.

#### **FR-2: Configuration**

  * **FR-2.1:** All project-specific configuration **must** be managed in a central `configs/settings.yaml` file.
  * **FR-2.2:** Settings **must** be loaded into a type-safe **Pydantic** model (`src/config.py`) for validated, IDE-friendly access.

#### **FR-3: Training Workflow (Local & Cloud)**

  * **FR-3.1:** The primary ML workflow **must** be built on the **Hugging Face** ecosystem (`transformers`, `datasets`, `accelerate`).
  * **FR-3.2:** The template **must** include a concrete, runnable example: fine-tuning a `distilbert-base-uncased` model on the `imdb` dataset.
  * **FR-3.3:** **Local Training:** The `train-local` task **must** use `accelerate launch` to enable seamless training on any local hardware (CPU, Mac MPS, single GPU).
  * **FR-3.4:** **Cloud Training:** The template **must** include a `sky_task.yaml` file to declaratively define a cloud training job for **SkyPilot**.
  * **FR-3.5:** The `train-cloud` task **must** use `sky launch` to execute the defined job on the user's chosen cloud provider.

#### **FR-4: Deployment Workflow**

  * **FR-4.1:** The template **must** include functionality to serve the fine-tuned model as a high-performance REST API.
  * **FR-4.2:** Model serving **must** be implemented using **LitServe**.
  * **FR-4.3:** A `serve` task **must** be provided to start the API server with a single command.

-----

### \#\# 5. Project Structure (Bill of Materials)

```
.
├── .github/
│   └── workflows/
│       └── ci.yaml
├── configs/
│   └── settings.yaml
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── deployment/
│   │   └── serve.py
│   └── models/
│       └── train_model.py
├── tests/
├── .gitignore
├── sky_task.yaml
├── pyproject.toml
└── README.md
```

-----

### \#\# 6. Out of Scope

  * **Heavyweight Frameworks:** Integration with complex distributed computing frameworks like **Ray** is intentionally excluded to maintain the template's simplicity and quick-start nature.
  * **Data Version Control:** DVC is excluded in favor of sourcing data directly from versioned repositories like the Hugging Face Hub.
  * **Experiment Tracking UI:** The template does not include a dedicated UI-based experiment tracker like MLflow or W\&B to keep the dependency footprint minimal. `tracelet` was considered but removed in favor of the `transformers.Trainer` logging capabilities.

-----

### \#\# 7. Success Metrics

  * **Time-to-First-Train:** A new user can successfully run `uv run train-local` within 5 minutes of generating the project.
  * **Time-to-Cloud:** A user can successfully launch a cloud job via `uv run train-cloud` with minimal setup (i.e., only configuring cloud credentials).
  * **Adoption:** The number of GitHub stars, forks, and active projects using the template.
