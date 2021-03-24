FROM harbor.apulis.cn:8443/algorithm/apulistech/apulisvision:2.0.0
RUN pip install kfserving==0.3.0
ENTRYPOINT ["python", "tools/model2pickle_kfserving.py"]

