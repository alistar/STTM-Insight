# Dockerfile for building streamline app

# base image
FROM python:3.7

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
    [general]\n\
    email = \"\"\n\
    " > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
    [server]\n\
    enableCORS = false\n\
    " > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

# copy local files into container
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "model/sttm_streamlit.py"]