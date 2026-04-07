FROM python:3.9

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user . $HOME/app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Command to run the app using gunicorn
# Note: 7860 is the default port Hugging Face Spaces listens on
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:server"]
