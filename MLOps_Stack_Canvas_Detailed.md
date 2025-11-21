# MLOps Stack Canvas
## Chest X-ray Multi-Disease Classification with DenseNet121 (Deep Learning)

---

### **PROJECT NAME:**
Chest X-ray Multi-Disease Classification with DenseNet121

### **DATE:**
2025-11-21

### **TEAM MEMBERS:**
- Name 1: [ใส่ชื่อสมาชิก 1]
- Name 2: [ใส่ชื่อสมาชิก 2]
- Name 3: [ใส่ชื่อสมาชิก 3]
- Name 4: [ใส่ชื่อสมาชิก 4]

---

## 🎯 1. VALUE PROPOSITION
### *Keep solving the real problem*

#### **🏥 Primary Value**
- **Multi-Disease Detection**: ตรวจจับโรคในภาพ X-ray จำนวน **15 ชนิด** เช่น:
    - Atelectasis (ปอดยุบ)
    - Cardiomegaly (หัวใจโต)
    - Consolidation (การแข็งตัวของปอด)
    - Edema (บวมน้ำ)
    - Effusion (น้ำในเยื่อหุ้มปอด)
    - Emphysema (ถุงลมโป่งพอง)
    - Fibrosis (เส้นใยในปอด)
    - Hernia (เฮอร์เนีย)
    - Infiltration (การแทรกซึม)
    - Mass (ก้อน)
    - No Finding (ปกติ)
    - Nodule (ปม)
    - Pleural Thickening (เยื่อหุ้มปอดหนา)
    - Pneumonia (ปอดอักเสบ)
    - Pneumothorax (ลมรั่วในช่องอก)

#### **⚡ Clinical Benefits**
- **Time Efficiency**: ช่วยแพทย์ลดเวลาในการวิเคราะห์ภาพ X-ray
- **Error Reduction**: ลดความผิดพลาดจาก human fatigue
- **24/7 Availability**: รองรับการใช้งานต่อเนื่อง

#### **🔍 Explainability**
- **Grad-CAM Integration**: มี Grad-CAM เพื่อช่วยอธิบายผลลัพธ์ (Explainability)
- **Clinical Trust**: เหมาะสำหรับการใช้งานจริงทางคลินิก
- **PACS Integration**: รองรับการใช้งานในโรงพยาบาลหรือระบบ PACS

---

## 📊 2. DATA SOURCES & DATA VERSIONING

#### **📁 Primary Dataset**
```
Dataset: NIH Chest X-ray Dataset
├── Total Images: 112,120 images
├── Format: .jpg files
├── Labels: .csv metadata
└── Classes: 15 disease categories
```

#### **🗂️ Data Organization**
- **Source**: NIH Chest X-ray Dataset (ChestX-ray14)
- **Format**: High-resolution JPEG images + CSV labels
- **Size**: ~42GB total dataset

#### **📈 Data Versioning Strategy**
```
Data Pipeline:
Raw Data → Cleaned Data → Augmented Data → Training Ready
    ↓           ↓             ↓              ↓
   v1.0       v1.1          v1.2           v1.3
```

**Tools & Methods:**
- **Git LFS**: สำหรับ large file versioning
- **DVC (Data Version Control)**: สำหรับ dataset versioning
- **Metadata Tracking**: บันทึก resolution, preprocessing steps
- **Data Lineage**: track data transformation history

#### **📋 Data Quality Assurance**
- Image integrity validation
- Resolution consistency check
- Label quality verification
- Missing data handling

**Implementation:**
- ใช้ `src/check_images.py` ตรวจสอบความสมบูรณ์ของรูปภาพ (missing, corrupted, invalid extension)
- สร้าง clean CSV อัตโนมัติ (archive/clean_labels.csv)
- รายงานปัญหาใน image_validation_report.txt

---

## 🔬 3. DATA ANALYSIS & EXPERIMENT MANAGEMENT

#### **📊 Exploratory Data Analysis**
```python
# Analysis Tools
├── Jupyter Notebook
├── pandas (data manipulation)
├── matplotlib + seaborn (visualization)
├── numpy (numerical operations)
└── PIL (image processing)
```

#### **🎯 Key Analysis Areas**
- **Class Distribution**: วิเคราะห์ class imbalance ของโรคแต่ละชนิด
- **Image Quality**: ตรวจสอบ blur, noise, artifacts
- **Resolution Analysis**: ความหลากหลายของขนาดภาพ
- **Medical Annotations**: ความถูกต้องของ ground truth

#### **🧪 Experiment Tracking**
**Primary Tools:**
- **Weights & Biases (wandb)**: สำหรับ comprehensive experiment tracking
- **TensorBoard**: สำหรับ real-time monitoring
- **MLflow**: สำหรับ model lifecycle management

**Implementation:**
- Mixed Precision Training (TensorFlow)
- Automatic Model Checkpointing (best_weights.weights.h5)
- Real-time Training Metrics (F1-score, AUC, Loss)
- Grad-CAM Visualization integrated in prediction

**Tracked Parameters:**
```yaml
Hyperparameters:
  - learning_rate: [1e-4, 5e-4, 1e-3]
  - optimizer: [Adam, AdamW, SGD]
  - batch_size: [16, 32, 64]
  - augmentation_strategy: [basic, advanced, custom]

Model Variants:
  - DenseNet121, DenseNet169
  - ResNet50, ResNet101
  - EfficientNet-B0 to B7

Metrics Tracked:
  - Overall Accuracy
  - Per-class F1-score
  - AUC-ROC per disease
  - Training/Validation Loss
  - Inference Time
```

---

## 🏭 4. FEATURE STORE & WORKFLOWS
### *Keep feature computation consistent along the ML lifecycle*

#### **🔄 Preprocessing Pipeline**
```python
# Shared Feature Workflow
preprocessing_pipeline = {
    'resize': (224, 224),
    'normalize': ImageNet_stats,
    'augmentation': {
        'RandomRotation': 15,
        'RandomHorizontalFlip': 0.5,
        'ColorJitter': (0.1, 0.1, 0.1, 0.1),
        'RandomAffine': 10
    }
}
```

#### **🛠️ Consistent Feature Engineering**
- **Torchvision Transforms**: standardized preprocessing
- **Reproducible Parameters**: ทุก environment ใช้ config เดียวกัน
- **Version Control**: preprocessing steps มี versioning
- **Testing**: unit tests สำหรับ feature pipeline

#### **📦 Feature Store Implementation**
- **Configuration Management**: YAML/JSON configs
- **Pipeline Orchestration**: reproducible workflows
- **Quality Gates**: automated feature validation

---

## 🏗️ 5. FOUNDATIONS
### *Reflecting DevOps*

#### **📝 Version Control Strategy**
```
Repository Structure:
├── src/               # Source code
├── configs/           # Configuration files
├── tests/             # Unit tests
├── docker/            # Docker configurations
├── .github/workflows/ # CI/CD pipelines
└── docs/              # Documentation
```

**Branching Strategy:**
```
main (production)
├── staging (pre-production)
├── develop (integration)
└── feature/* (development)
```

#### **⚙️ Development Environment**
- **GitHub/GitLab**: primary version control
- **Code Quality**: 
  - **flake8**: linting
  - **black**: code formatting
  - **mypy**: type checking
- **Testing Framework**: **pytest** for unit tests
- **Pre-commit Hooks**: automated quality checks

**Troubleshooting & Resource Management:**
- คู่มือแก้ปัญหา GPU, memory, module error (ดู README.md section troubleshooting)
- ปรับ batch size, learning rate, epochs ได้ใน train.py

#### **🖥️ Infrastructure Requirements**
```
Hardware Specifications:
├── Training: GPU-enabled (RTX 3080+ or V100+)
├── Inference: CPU/GPU hybrid
├── Storage: SSD for fast I/O
└── Memory: 32GB+ RAM recommended

Deployment Options:
├── Local Development: Docker containers
├── Cloud Training: Google Colab Pro, AWS EC2
└── Production: Kubernetes clusters
```

---

## 🔄 6. CI/CT/CD: ML PIPELINE ORCHESTRATION
### *Building, testing, packaging and deploying ML pipeline*

#### **🔨 Continuous Integration (CI)**
```yaml
# GitHub Actions Workflow
CI Pipeline:
  - Code Quality Check:
    ├── flake8 (linting)
    ├── black (formatting)
    ├── mypy (type checking)
    └── pytest (unit tests)
  
  - Data Validation:
    ├── Schema validation
    ├── Data integrity checks
    └── Feature pipeline tests
  
  - Model Testing:
    ├── Model architecture tests
    ├── Training pipeline validation
    └── Inference functionality
```

#### **🎯 Continuous Training (CT)**
```python
# Automated Retraining Triggers
retraining_conditions = {
    'data_drift': threshold_0.1,
    'performance_degradation': accuracy_drop_5_percent,
    'new_data_volume': min_1000_samples,
    'scheduled': weekly_retraining
}
```

#### **🚀 Continuous Deployment (CD)**
```
Deployment Pipeline:
Raw Model → Model Validation → Artifact Creation → Registry Push → Deployment

Tools Stack:
├── GitHub Actions: automation
├── DVC Pipelines: ML workflow orchestration
├── Docker: containerization
├── FastAPI: model serving
└── Kubernetes: scaling (production)
```

---

## 📦 7. MODEL REGISTRY & MODEL VERSIONING

#### **🏛️ Model Registry Strategy**
```
MLflow Model Registry Structure:
├── Production Models:
│   ├── best_accuracy_v2.1.pt
│   ├── best_auc_v2.0.pt
│   └── current_production.pt
├── Staging Models:
│   ├── candidate_v2.2.pt
│   └── experimental_v3.0.pt
└── Archive:
    ├── baseline_v1.0.pt
    └── lightweight_edge_v1.5.pt
```

#### **📊 Model Metadata Tracking**
```json
{
  "model_version": "v2.1",
  "training_config": {
    "architecture": "DenseNet121",
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 50
  },
  "dataset_version": "v1.3",
  "performance_metrics": {
    "overall_accuracy": 0.847,
    "mean_auc": 0.912,
    "per_class_f1": {...}
  },
  "training_time": "2.5 hours",
  "model_size": "28.7 MB"
}
```

#### **🔄 Model Lifecycle Management**
- **Automatic Promotion**: staging → production based on metrics
- **Rollback Strategy**: instant fallback to previous stable version
- **A/B Testing**: gradual rollout of new models
- **Model Retirement**: systematic deprecation process

**Artifact Storage:**
- Model weights: `saved_models/best_weights.weights.h5`
- Training logs & metrics: `saved_models/` และ log files

---

## 🚀 8. MODEL DEPLOYMENT

#### **🌐 Deployment Architecture**
```
Production Deployment Stack:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Load        │    │ API Gateway  │    │ Model       │
│ Balancer    │───▶│ (FastAPI)    │───▶│ Service     │
│ (Nginx)     │    │              │    │ (GPU/CPU)   │
└─────────────┘    └──────────────┘    └─────────────┘
       │                    │                   │
       ▼                    ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Monitoring  │    │ Logging      │    │ Model       │
│ (Prometheus)│    │ (ELK Stack)  │    │ Storage     │
└─────────────┘    └──────────────┘    └─────────────┘
```

#### **🔧 Deployment Options**

**Option 1: REST API (Recommended)**
```python
# FastAPI Implementation
@app.post("/predict")
async def predict_disease(image: UploadFile):
    # Process image
    # Run inference
    # Return results + confidence scores

@app.post("/gradcam")
async def generate_gradcam(image: UploadFile, target_class: str):
    # Generate interpretability visualization
    # Return heatmap overlay
```

**Option 2: Containerized Deployment**
```dockerfile
# Multi-stage Docker build
FROM pytorch/pytorch:1.12-cuda11.3-cudnn8-runtime
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Option 3: Cloud Deployment**
- **AWS**: ECS/EKS with GPU support
- **GCP**: Cloud Run with custom containers
- **Azure**: Container Instances with GPU

#### **⚡ Scaling Strategy**
- **Horizontal Scaling**: Kubernetes auto-scaling
- **Load Balancing**: distribute requests across replicas
- **GPU Optimization**: efficient GPU memory management
- **Caching**: Redis for frequent predictions

---

## 🎯 9. PREDICTION SERVING

#### **🔌 API Endpoints**
```python
# Primary Inference Endpoint
POST /api/v1/predict
{
    "image": "base64_encoded_xray_image",
    "return_probabilities": true,
    "threshold": 0.5
}

Response:
{
    "predictions": {
        "Pneumonia": 0.85,
        "Mass": 0.12,
        "Nodule": 0.08,
        ...
    },
    "processing_time": 150,  # milliseconds
    "model_version": "v2.1"
}

# Interpretability Endpoint
POST /api/v1/gradcam
{
    "image": "base64_encoded_xray_image",
    "target_classes": ["Pneumonia", "Mass"]
}

Response:
{
    "gradcam_images": {
        "Pneumonia": "base64_encoded_heatmap",
        "Mass": "base64_encoded_heatmap"
    },
    "processing_time": 300
}
```

#### **⚡ Performance Requirements**
- **Response Time**: < 200ms (with GPU)
- **Throughput**: 100+ requests/minute
- **Availability**: 99.9% uptime
- **Scalability**: auto-scale based on load

#### **🛡️ API Features**
- **Authentication**: API key management
- **Rate Limiting**: prevent abuse
- **Input Validation**: image format/size checks
- **Error Handling**: comprehensive error responses
- **Monitoring**: request/response logging

---

## 📊 10. MODEL & DATA & APPLICATION MONITORING

#### **🎯 Performance Monitoring**
```python
# Key Metrics Dashboard
monitoring_metrics = {
    'Model Performance': {
        'accuracy_drift': 'weekly_comparison',
        'prediction_confidence': 'distribution_tracking',
        'class_distribution': 'input_data_monitoring'
    },
    'System Performance': {
        'response_time': 'p95_latency',
        'throughput': 'requests_per_minute',
        'error_rate': 'percentage_failed_requests'
    },
    'Infrastructure': {
        'gpu_utilization': 'percentage',
        'memory_usage': 'gb_consumed',
        'disk_space': 'storage_available'
    }
}
```

#### **🚨 Drift Detection**
**Data Drift Monitoring:**
- **Image Quality**: brightness, contrast distribution changes
- **Input Distribution**: new patient demographics
- **Technical Drift**: different X-ray machines/protocols

**Model Drift Detection:**
- **Performance Degradation**: accuracy drop > 5%
- **Confidence Score Changes**: unusual prediction patterns
- **Class Imbalance Shifts**: changing disease prevalence

#### **🛠️ Monitoring Tools Stack**
```
Monitoring Infrastructure:
├── Prometheus: metrics collection
├── Grafana: visualization dashboards
├── MLflow: ML-specific monitoring
├── ELK Stack: log analysis
└── AlertManager: notification system

Alert Conditions:
├── Accuracy drop > 5%
├── Response time > 500ms
├── Error rate > 1%
└── GPU utilization > 90%
```

#### **📊 Real-time Dashboards**
- **Clinical Dashboard**: per-disease accuracy trends
- **Technical Dashboard**: system performance metrics
- **Business Dashboard**: usage statistics and ROI

---

## ⚠️ 0. MLOPS DILEMMAS

#### **🎯 Technical Challenges**

**Dataset Imbalance Crisis:**
```
Disease Distribution Problem:
├── Common: Infiltration (19%), No Finding (60%)
├── Rare: Pneumothorax (5%), Hernia (0.2%)
└── Impact: Low recall for rare diseases

Solutions Implemented:
├── Weighted Loss Functions
├── Focal Loss for hard examples
├── SMOTE-like augmentation
└── Ensemble methods
```

**Image Quality Variability:**
- **Resolution Inconsistency**: 1024x1024 to 4892x4020 pixels
- **Equipment Variations**: different X-ray machines
- **Preprocessing Robustness**: need adaptive normalization

**Interpretability Reliability:**
- **Grad-CAM Limitations**: sometimes highlights irrelevant regions
- **Clinical Validation**: need radiologist confirmation
- **False Confidence**: high predictions on uncertain cases

#### **🔒 Data Privacy & Compliance**
```
HIPAA Compliance Requirements:
├── Data Anonymization: remove all PHI
├── Secure Storage: encrypted data at rest
├── Access Control: role-based permissions
├── Audit Logging: track all data access
└── Data Retention: automatic deletion policies
```

#### **💰 Resource Optimization**
**GPU Cost Management:**
- **Training Costs**: $50-200 per full training run
- **Inference Optimization**: batch processing for efficiency
- **Auto-scaling**: cost-effective resource allocation
- **Model Compression**: reduce computational requirements

#### **🔄 Operational Challenges**
- **Model Staleness**: when to retrain?
- **Version Management**: seamless model updates
- **Downtime Management**: zero-downtime deployments
- **Performance Degradation**: early warning systems

---

## 🗄️ 11. METADATA STORE

#### **📊 Comprehensive Metadata Management**
```json
{
  "metadata_categories": {
    "data_versions": {
      "dataset_v1.3": {
        "source": "NIH_ChestX-ray14",
        "size": "112,120_images",
        "preprocessing": "resize_224x224_normalize_imagenet",
        "splits": "train_70_val_15_test_15",
        "creation_date": "2025-11-15"
      }
    },
    "model_versions": {
      "densenet121_v2.1": {
        "architecture": "DenseNet121_pretrained",
        "fine_tuning": "full_model",
        "hyperparameters": {
          "learning_rate": 1e-4,
          "batch_size": 32,
          "optimizer": "AdamW",
          "scheduler": "CosineAnnealingLR"
        },
        "training_duration": "2.5_hours",
        "gpu_used": "NVIDIA_RTX_3080"
      }
    },
    "experiment_configs": {
      "exp_042": {
        "objective": "improve_rare_disease_recall",
        "modifications": "focal_loss_alpha_0.75",
        "results": "improved_pneumothorax_recall_15_percent"
      }
    },
    "evaluation_results": {
      "per_class_metrics": {
        "Pneumonia": {"precision": 0.89, "recall": 0.85, "f1": 0.87},
        "Mass": {"precision": 0.78, "recall": 0.72, "f1": 0.75},
        "Nodule": {"precision": 0.82, "recall": 0.79, "f1": 0.80}
      },
      "overall_metrics": {
        "macro_avg_f1": 0.81,
        "weighted_avg_f1": 0.84,
        "mean_auc_roc": 0.91
      }
    }
  }
}
```

#### **🛠️ Metadata Tools & Storage**
```
Metadata Infrastructure:
├── MLflow Tracking: experiment metadata
├── DVC: data and model versioning metadata
├── PostgreSQL: structured metadata storage
├── MongoDB: unstructured experiment notes
└── Git: code and config versioning

**Contribution Workflow:**
- Fork, Branch, Pull Request ตามขั้นตอนใน README.md
```

---

## 📁 Project Structure (implementation)
```
train-LLM-Chest-X-rays/
├── src/
│   ├── dataset.py          # Data handling
│   ├── model.py           # DenseNet121 model
│   ├── train.py           # Training pipeline
│   ├── predict.py         # Prediction & Grad-CAM
│   ├── check_images.py    # Image validation tool
│   └── utils.py           # Utilities
├── archive/               # Training data
│   ├── new_labels.csv     # Labels
│   └── resized_images/    # Images
├── saved_models/          # Trained model weights
├── predict_images/        # Images for prediction
└── requirements.txt       # Dependencies
```

---

### 📋 **Quick Reference Summary**

| **Component** | **Tool/Technology** | **Purpose** |
|--------------|-------------------|------------|
| **Data Versioning** | DVC + Git LFS | Dataset version control |
| **Experiment Tracking** | Weights & Biases | Hyperparameter tuning |
| **Model Registry** | MLflow | Model lifecycle management |
| **Deployment** | FastAPI + Docker | Model serving |
| **Monitoring** | Prometheus + Grafana | Performance tracking |
| **CI/CD** | GitHub Actions | Automation pipeline |

---

*This MLOps Stack Canvas provides a comprehensive blueprint for implementing a production-ready chest X-ray classification system with proper MLOps practices and governance.*