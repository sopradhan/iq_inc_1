import sqlite3
import json
import random
import uuid
import datetime

DB_PATH = "D:/incident_management/data/sqlite/incident_management_v2.db"

azure_services = [
    "Storage", "Sql", "KeyVault", "Network", "AppService",
    "Compute", "Monitoring", "EventHub", "Functions"
]

# Azure resource types typical for services
resource_types = {
    "Storage": "Microsoft.Storage/storageAccounts",
    "Sql": "Microsoft.Sql/servers/databases",
    "KeyVault": "Microsoft.KeyVault/vaults",
    "Network": "Microsoft.Network/virtualNetworks",
    "AppService": "Microsoft.Web/sites",
    "Compute": "Microsoft.Compute/virtualMachines",
    "Monitoring": "Microsoft.Insights/components",
    "EventHub": "Microsoft.EventHub/namespaces",
    "Functions": "Microsoft.Web/sites/functions"
}

metrics_info = [
    {"metricName": "Http5xx", "message": "api call failed with status 500", "threshold": 5},
    {"metricName": "Http4xx", "message": "api call rate limited with status 429", "threshold": 10},
    {"metricName": "ResponseTime", "message": "api response time exceeded threshold", "threshold": 200},
    {"metricName": "SuccessE2ELatency", "message": "end to end success latency high", "threshold": 150},
]

incident_errors = [
    "authentication failed for user",
    "database connection timeout",
    "resource quota exceeded",
    "DNS resolution failed",
    "api gateway unreachable",
    "storage blob not found",
    "failed to acquire lock on resource",
    "configuration inconsistency detected",
]

def load_severity_rules():
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute("""
        SELECT pattern, severity_level, base_score, category, description, environment 
        FROM severity_rules
        """)
        rows = cursor.fetchall()
        cursor.close()
        rules = []
        for r in rows:
            rules.append({
                "pattern": r[0],
                "severity_level": r[1],
                "base_score": r[2],
                "category": r[3],
                "description": r[4],
                "environment": r[5]
            })
        return rules

def generate_metrics_log_entry(rule, service):
    metric = random.choice(metrics_info)
    threshold = metric["threshold"]

    severity = rule["severity_level"]

    if severity == "S1":
        value = threshold + random.uniform(5, 20)
    elif severity == "S2":
        value = threshold + random.uniform(2, 5)
    elif severity == "S3":
        value = threshold + random.uniform(0.5, 2)
    else:
        value = threshold - random.uniform(0, 0.5)

    timestamp = (datetime.datetime.utcnow() - datetime.timedelta(days=random.randint(0, 30))).isoformat() + "Z"
    resource_type = resource_types.get(service, "Unknown")

    entry = {
        "id": str(uuid.uuid4()),
        "type": "metrics_log",
        "azure_service": service,
        "resource_type": resource_type,
        "metricName": metric["metricName"],
        "value": round(value, 2),
        "severity_level": severity,
        "description": metric["message"],
        "category": rule["category"],
        "environment": rule["environment"],
        "timestamp": timestamp,
        "pattern": metric["message"],
        "keywords": metric["metricName"] + " " + metric["message"],
        "key_metrics": rule["base_score"]
    }
    return json.dumps(entry)

def generate_incident_log_entry(rule, service):
    error_msg = random.choice(incident_errors)
    timestamp = (datetime.datetime.utcnow() - datetime.timedelta(days=random.randint(0, 30))).isoformat() + "Z"
    status_map = {"S1": "Failed", "S2": "Failed", "S3": "Warning", "S4": "Success"}
    level_map = {"S1": "Error", "S2": "Error", "S3": "Warning", "S4": "Informational"}
    resource_type = resource_types.get(service, "Unknown")

    entry = {
        "id": str(uuid.uuid4()),
        "type": "incident_log",
        "azure_service": service,
        "resource_type": resource_type,
        "error_message": error_msg,
        "severity_level": rule["severity_level"],
        "category": rule["category"],
        "description": rule["description"],
        "environment": rule["environment"],
        "status": status_map.get(rule["severity_level"], "Failed"),
        "level": level_map.get(rule["severity_level"], "Error"),
        "timestamp": timestamp,
        "keywords": error_msg,
        "pattern": error_msg,
        "key_metrics": rule["base_score"]
    }
    return json.dumps(entry)

def generate_synthetic_training_data(num_samples=2000):
    rules = load_severity_rules()
    rules_by_severity = {}
    for rule in rules:
        rules_by_severity.setdefault(rule["severity_level"], []).append(rule)

    severity_levels = list(rules_by_severity.keys())
    samples_per_level = max(num_samples // len(severity_levels), 1)

    data = []
    for sev in severity_levels:
        level_rules = rules_by_severity[sev]
        for _ in range(samples_per_level):
            rule = random.choice(level_rules)
            service = random.choice(azure_services)
            if random.random() > 0.5:
                data.append(generate_metrics_log_entry(rule, service))
            else:
                data.append(generate_incident_log_entry(rule, service))
    return data

def save_data_to_db(data):
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute("""
        DROP TABLE IF EXISTS synthetic_training_data;
        """)
        cursor.execute("""
        CREATE TABLE synthetic_training_data (
            id TEXT PRIMARY KEY,
            json_payload TEXT NOT NULL,
            severity_level TEXT NOT NULL,
            azure_service TEXT,
            resource_type TEXT,
            keywords TEXT,
            key_metrics REAL
        )
        """)
        for record in data:
            obj = json.loads(record)
            cursor.execute("""
            INSERT OR IGNORE INTO synthetic_training_data 
            (id, json_payload, severity_level, azure_service, resource_type, keywords, key_metrics) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (obj["id"], record, obj["severity_level"], obj.get("azure_service"), obj.get("resource_type"), obj.get("keywords"), obj.get("key_metrics")))
        conn.commit()

if __name__ == "__main__":
    synthetic_data = generate_synthetic_training_data(2000)
    save_data_to_db(synthetic_data)
    print("Synthetic training data with Azure service and resource type created and saved.")
