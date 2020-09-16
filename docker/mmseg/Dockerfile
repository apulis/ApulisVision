# Install mmsegmentation
FROM mmdetection:latest
RUN git clone https://github.com/open-mmlab/mmsegmenation.git /mmsegmentation
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
