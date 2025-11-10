import sqlite3
import json
import uuid
import random
from datetime import datetime, timedelta
import ipaddress
from pathlib import Path

# Database configuration
DB_PATH = "D:/incident_management/data/sqlite/incident_management_v3.db"

class AzureEnvironmentConfig:
    def __init__(self):
        # Environment configuration
        self.environments = {
            "prod": {
                "subscription_id": str(uuid.uuid4()),
                "subscription_name": "Production",
                "allowed_regions": ["eastus", "westeurope", "southeastasia"],
                "resource_prefix": "prod",
                "tags": {
                    "Environment": "Production",
                    "CostCenter": "IT-Production",
                    "DataClassification": "Confidential",
                    "BusinessCriticality": "Mission-Critical"
                }
            },
            "uat": {
                "subscription_id": str(uuid.uuid4()),
                "subscription_name": "UAT",
                "allowed_regions": ["eastus2", "westus2"],
                "resource_prefix": "uat",
                "tags": {
                    "Environment": "UAT",
                    "CostCenter": "IT-Testing",
                    "DataClassification": "Internal",
                    "BusinessCriticality": "Important"
                }
            },
            "dev": {
                "subscription_id": str(uuid.uuid4()),
                "subscription_name": "Development",
                "allowed_regions": ["eastus2", "westus2"],
                "resource_prefix": "dev",
                "tags": {
                    "Environment": "Development",
                    "CostCenter": "IT-Development",
                    "DataClassification": "Internal",
                    "BusinessCriticality": "Low"
                }
            }
        }

        # Service configurations
        self.services = {
            "Microsoft.KeyVault": {
                "type": "vaults",
                "operations": ["VaultGet", "KeyGet", "KeyCreate", "SecretGet", "SecretSet"],
                "resultTypes": ["Success", "Failed"],
                "metrics": ["ServiceApiLatency", "SaturationShoebox", "Availability"],
                "diagnostic_categories": ["AuditEvent", "AzurePolicyEvaluation"]
            },
            "Microsoft.Sql": {
                "type": "servers/databases",
                "operations": ["DatabaseConnect", "QueryExecute", "BackupComplete", "DatabaseFailover"],
                "resultTypes": ["Succeeded", "Failed"],
                "metrics": ["cpu_percent", "storage_percent", "dtu_consumption_percent"],
                "diagnostic_categories": ["SQLSecurityAuditEvents", "AutomaticTuning"]
            },
            "Microsoft.Web": {
                "type": "sites",
                "operations": ["AppServicePlanUpdate", "WebAppRestart", "SiteConfigUpdate"],
                "resultTypes": ["Succeeded", "Failed", "InProgress"],
                "metrics": ["Http5xx", "ResponseTime", "CpuTime"],
                "diagnostic_categories": ["AppServiceHTTPLogs", "AppServiceConsoleLogs"]
            },
            "Microsoft.Storage": {
                "type": "storageAccounts",
                "operations": ["BlobGet", "BlobCreate", "ContainerDelete", "StorageRead"],
                "resultTypes": ["Success", "Failed"],
                "metrics": ["Availability", "Transactions", "SuccessE2ELatency"],
                "diagnostic_categories": ["StorageRead", "StorageWrite", "StorageDelete"]
            }
        }

        # Error patterns by environment
        self.error_patterns = {
            "prod": {
                "critical": [
                    "High Availability Failover Initiated",
                    "Database Deadlock Detected",
                    "SSL Certificate Expiration Critical",
                    "Memory Resource Exhaustion"
                ],
                "high": [
                    "Elevated Error Rate Detected",
                    "Network Connectivity Issues",
                    "Database Performance Degradation"
                ]
            },
            "uat": {
                "critical": [
                    "Test Failover Simulation",
                    "Load Test Resource Exhaustion",
                    "Integration Test Failure"
                ],
                "high": [
                    "Performance Test Threshold Breach",
                    "API Integration Failure",
                    "Data Sync Issues"
                ]
            },
            "dev": {
                "critical": [
                    "Development Environment Down",
                    "Build Pipeline Failure",
                    "Development Database Corruption"
                ],
                "high": [
                    "Development API Gateway Issues",
                    "Local Development Stack Error",
                    "Test Data Generation Failure"
                ]
            }
        }

    def get_resource_name(self, service, env):
        prefix = self.environments[env]["resource_prefix"]
        service_short = service.split('.')[-1].lower()
        return f"{prefix}-{service_short}-{random.randint(1,999):03d}"

    def get_resource_id(self, service, env):
        subscription_id = self.environments[env]["subscription_id"]
        region = random.choice(self.environments[env]["allowed_regions"])
        rg_name = f"{self.environments[env]['resource_prefix']}-rg-{region}"
        resource_name = self.get_resource_name(service, env)
        return f"/subscriptions/{subscription_id}/resourceGroups/{rg_name}/providers/{service}/{self.services[service]['type']}/{resource_name}"

class IncidentLogGenerator:
    def __init__(self):
        self.config = AzureEnvironmentConfig()
        
    def generate_activity_log(self, env):
        service = random.choice(list(self.config.services.keys()))
        operation = random.choice(self.config.services[service]["operations"])
        resource_id = self.config.get_resource_id(service, env)
        status = random.choice(self.config.services[service]["resultTypes"])
        
        log = {
            "correlationId": str(uuid.uuid4()),
            "eventTimestamp": (datetime.utcnow() - timedelta(minutes=random.randint(0, 1440))).isoformat() + "Z",
            "category": "Administrative",
            "resourceId": resource_id,
            "operationName": {
                "value": operation,
                "localizedValue": operation
            },
            "status": {
                "value": status,
                "localizedValue": status
            },
            "subscriptionId": resource_id.split('/')[2],
            "tags": self.config.environments[env]["tags"],
            "properties": {
                "statusCode": 200 if status == "Success" else 500,
                "serviceRequestId": str(uuid.uuid4()),
                "eventCategory": "Administrative",
                "environment": env.upper()
            }
        }

        if status != "Success":
            error_type = "critical" if random.random() < 0.3 else "high"
            error_message = random.choice(self.config.error_patterns[env][error_type])
            log["properties"]["error"] = {
                "code": f"{error_type.upper()}_ERROR",
                "message": error_message
            }

        return log

    def generate_metric_log(self, env):
        service = random.choice(list(self.config.services.keys()))
        resource_id = self.config.get_resource_id(service, env)
        metric_name = random.choice(self.config.services[service]["metrics"])
        
        # Generate appropriate metric value based on environment
        base_value = random.uniform(0, 100)
        if env == "prod":
            # Production: More stable metrics
            value = base_value * 0.7  # 70% of max
        elif env == "uat":
            # UAT: More variable metrics
            value = base_value * random.uniform(0.4, 0.9)
        else:
            # Dev: Highly variable metrics
            value = base_value * random.uniform(0.2, 1.0)

        return {
            "time": datetime.utcnow().isoformat() + "Z",
            "resourceId": resource_id,
            "metricName": metric_name,
            "timeGrain": "PT1M",
            "value": round(value, 2),
            "tags": self.config.environments[env]["tags"],
            "properties": {
                "environment": env.upper(),
                "subscription": self.config.environments[env]["subscription_name"],
                "metric_category": "Platform",
                "unit": "Percent" if "percent" in metric_name.lower() else "Count"
            }
        }

def generate_mock_data(total_records=1000):
    generator = IncidentLogGenerator()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Environment distribution (50% prod, 30% uat, 20% dev)
    env_distribution = {
        "prod": int(total_records * 0.5),
        "uat": int(total_records * 0.3),
        "dev": int(total_records * 0.2)
    }

    print("Generating mock data with environment distribution:")
    for env, count in env_distribution.items():
        print(f"{env.upper()}: {count} records")

    # Clear existing data
    cursor.execute("DELETE FROM incident_logs")

    records_created = 0
    for env, count in env_distribution.items():
        for i in range(count):
            # 70% activity logs, 30% metric logs
            if random.random() < 0.7:
                log = generator.generate_activity_log(env)
                source_type = "ActivityLog"
            else:
                log = generator.generate_metric_log(env)
                source_type = "MetricLog"

            cursor.execute("""
                INSERT INTO incident_logs (id, incident_json, source_type, status)
                VALUES (?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                json.dumps(log),
                source_type,
                "new"
            ))

            records_created += 1
            if records_created % 100 == 0:
                print(f"Created {records_created} records...")

    conn.commit()

    # Print summary
    cursor.execute("""
    SELECT 
        json_extract(incident_json, '$.properties.environment') as env,
        source_type,
        COUNT(*) as count
    FROM incident_logs
    GROUP BY env, source_type
    """)
    
    print("\nFinal Distribution:")
    results = cursor.fetchall()
    for env, source_type, count in results:
        print(f"Environment: {env}, Type: {source_type}, Count: {count}")

    conn.close()
    print("\nMock data generation completed!")

if __name__ == "__main__":
    generate_mock_data(1000)