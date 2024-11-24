FROM python:3.10 as base


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy

RUN apt-get update && apt-get install -y --no-install-recommends gcc ffmpeg libsm6 libxext6 python3-opencv libgl1 libgl1-mesa-dev

FROM base AS runtime

COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /home/appuser
RUN pip install "fastapi[standard]"
USER appuser

# Install application into container
COPY . .

# Run the application
CMD ["fastapi", "run", "main.py", "--port", "80"]