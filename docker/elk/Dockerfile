FROM docker.elastic.co/logstash/logstash:8.13.4

USER root
RUN /usr/share/logstash/bin/logstash-plugin install logstash-input-google_pubsub \
    && mkdir -p /var/log/noesis \
    && chown -R logstash:logstash /var/log/noesis \
    && rm -f /usr/share/logstash/pipeline/*.conf
USER logstash

COPY pipeline.conf /usr/share/logstash/pipeline/pipeline.conf
COPY logstash.yml /usr/share/logstash/config/logstash.yml

VOLUME ["/var/log/noesis"]

