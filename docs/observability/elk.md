# ELK Stack für lokale Entwicklung

Die Elastic-Komponenten laufen in einem separaten Compose-Stack unter `docker/elk`. Die Konfiguration ist für Entwicklungszwecke gedacht und verzichtet auf TLS-Zertifikate, erfordert aber Dev-Passwörter.

## Voraussetzungen
- Docker und Docker Compose v2
- Ausreichend Arbeitsspeicher (mind. 4 GB RAM für Elasticsearch + Kibana)
- Ein Verzeichnis mit Anwendungs-Logs im JSON-Format (Standard: `logs/app/*.log` und `logs/app/chaos/*.json`)

## Starten
```bash
# Gesamtes Dev-Setup per npm (App + ELK + Seeding)
npm run dev:stack

# Gesamtes Dev-Setup (App + ELK) von Grund auf bauen & starten
bash scripts/dev-up-all.sh

# Nur den ELK-Stack starten (wenn App bereits läuft)
docker compose -f docker/elk/docker-compose.yml up -d

# ELK-Stack wieder stoppen
docker compose -f docker/elk/docker-compose.yml down
```

Das Oneshot-Skript `scripts/dev-up-all.sh` (alias `npm run dev:stack`) führt folgende Schritte aus:

1. Baut die lokalen Docker-Images für Anwendung und ELK-Stack.
2. Startet beide Compose-Stacks.
3. Wartet, bis der Web-Service reagiert, und führt `npm run dev:init` (Migrationen, Bootstrap) aus.
4. Seedet Demo- und Heavy-Datasets (`npm run seed:demo`, `npm run seed:heavy`).

Dabei legt es das Log-Verzeichnis (`APP_LOG_PATH`, Standard `logs/app`) automatisch an.

Standard-Credentials für Elasticsearch/Kibana liegen in [`../../.env.dev-elk`](../../.env.dev-elk). Das Skript lädt die Datei automatisch, sofern sie vorhanden ist.

Vor dem Start können folgende Variablen gesetzt werden:

| Variable | Zweck | Default |
| --- | --- | --- |
| `ELASTIC_PASSWORD` | Passwort für den `elastic`-User | `changeme` |
| `KIBANA_SYSTEM_PASSWORD` | Passwort für den `kibana_system`-Account | `changeme` |
| `APP_LOG_PATH` | Pfad zum Log-Verzeichnis der Anwendung | `../../logs/app` relativ zu `docker/elk` |
| `KIBANA_PUBLIC_URL` | Öffentliche URL (Proxy) für Kibana | `http://localhost:5601` |

Die Logs werden schreibgeschützt unter `/var/log/noesis` im Logstash-Container gemountet. Der Stack öffnet folgende Ports:

- `9200` für Elasticsearch (Basic Auth: `elastic` + `ELASTIC_PASSWORD`)
- `5601` für Kibana (Login: `elastic` + `ELASTIC_PASSWORD`, oder `kibana_system` per API)
- `5044` für Beats-Inputs (optional, z. B. Filebeat)

## Nutzung
1. Kibana ist nach dem Start unter [http://localhost:5601](http://localhost:5601) erreichbar. Melde dich mit dem `elastic`-Benutzer an.
2. Logstash liest lokale JSON-Logs (`timestamp`-Feld empfohlen) aus dem gemounteten Verzeichnis und schreibt sie in Indizes `noesis-app-*`. Chaos-Testläufe landen unter `logs/app/chaos/*.json` und werden mit dem Feld `test_suite=chaos` markiert, sodass Kibana-Discover-Abfragen wie `test_suite:chaos` die Reports filtern.
3. Für Filebeat-Setups kann `localhost:5044` als Ziel genutzt werden. Zertifikate müssen ggf. ergänzt werden.

## Betrieb in Google Cloud
Für produktionsnahe Tests nutzen wir die Google-Cloud-Logging-Pipeline als Quelle. Die Logstash-Konfiguration enthält dafür einen optionalen `google_pubsub`-Input. Vorgehen:

1. **Cloud Logging → Pub/Sub**
   ```bash
   gcloud logging sinks create noesis-elk \
     pubsub.googleapis.com/projects/$PROJECT_ID/topics/noesis-elk \
     --log-filter='resource.type=("cloud_run_revision" OR "k8s_container")'

   gcloud pubsub subscriptions create noesis-elk-sub \
     --topic=noesis-elk --ack-deadline=30
   ```
   Der Sink streamt Cloud-Run- bzw. GKE-Container-Logs in das Pub/Sub-Topic.

2. **Service Account für Logstash**
   ```bash
   gcloud iam service-accounts create logstash-subscriber --project $PROJECT_ID
   gcloud pubsub subscriptions add-iam-policy-binding noesis-elk-sub \
     --member="serviceAccount:logstash-subscriber@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/pubsub.subscriber"
   gcloud iam service-accounts keys create ~/Downloads/noesis-logstash.json \
     --iam-account logstash-subscriber@$PROJECT_ID.iam.gserviceaccount.com
   ```
   Die JSON-Datei bleibt außerhalb des Repos und wird später gemountet.

3. **Compose-Stack mit Pub/Sub-Empfang starten**
   ```bash
   export GCP_PROJECT_ID=$PROJECT_ID
   export GCP_PUBSUB_TOPIC=noesis-elk
   export GCP_PUBSUB_SUBSCRIPTION=noesis-elk-sub
   export GCP_CREDENTIALS_PATH=~/Downloads/noesis-logstash.json

   docker compose \
     -f docker/elk/docker-compose.yml \
     -f docker/elk/docker-compose.gcloud.yml \
     up
   ```
   Standardmäßig erwartet Logstash die Credentials unter `/usr/share/logstash/config/gcp-service-account.json`. Du kannst den Zielpfad mit `GCP_CREDENTIALS_FILE` überschreiben.

Die GCP-Pfade ergänzen die lokale Volume-Quelle; beide Inputs (Datei + Pub/Sub) können parallel aktiv sein. Für produktive Workloads sollten Pub/Sub-Acks überwacht und Retention/Dead-Letter-Queues konfiguriert werden.

## Bekannte Einschränkungen
- Elasticsearch benötigt ca. 4 GB RAM; auf schwächeren Maschinen kann der Dienst nicht starten.
- TLS ist deaktiviert. Für produktionsnahe Tests müssen Zertifikate und Transportverschlüsselung ergänzt werden.
- Standard-Logformat erwartet JSON mit einem `timestamp`-Feld. Andere Formate erfordern Anpassungen in `docker/elk/pipeline.conf`.
