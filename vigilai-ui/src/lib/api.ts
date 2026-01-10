const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:5431/api';

export interface TrainingParams {
  epochs: number;
  batch_size: number;
  img_size: number;
  model_size: string;
  training_type?: 'standard' | 'combined';
}

export interface DetectionParams {
  camera_index: number;
  conf_threshold: number;
  model_size: string;
  model_path?: string;
}

export interface EvaluationParams {
  model_path: string;
  evaluation_type?: 'standard' | 'combined';
}

export interface ModelInfo {
  name: string;
  path: string;
  modified: number;
}

export interface ModelDetails {
  model_path: string;
  model_type: string;
  model_size_mb: number;
  parameters: string;
  inference_speed: string;
  classes: string[];
  modified_time: number;
  can_evaluate_combined: boolean;
}

export interface EvaluationResults {
  mAP50?: number;
  mAP95?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  [key: string]: any;
}

export interface SystemStatus {
  status: string;
  detection_running: boolean;
  timestamp: string;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

export interface DatasetStatus {
  datasets: Record<string, boolean>;
  available_count: number;
  total_count: number;
  can_train_combined: boolean;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  getBaseUrl() {
    return this.baseUrl;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      return response.json();
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Network error: Unable to connect to server');
      }
      throw error;
    }
  }

  // System Status
  async getStatus(): Promise<SystemStatus> {
    return this.request<SystemStatus>('/status');
  }

  async getLogs(limit: number = 50): Promise<{ logs: LogEntry[]; total: number }> {
    return this.request<{ logs: LogEntry[]; total: number }>(`/logs?limit=${limit}`);
  }

  // Training
  async startTraining(params: TrainingParams): Promise<{ message: string; parameters: TrainingParams }> {
    return this.request<{ message: string; parameters: TrainingParams }>('/train', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // Detection
  async startDetection(params: DetectionParams): Promise<{ message: string; parameters: DetectionParams }> {
    return this.request<{ message: string; parameters: DetectionParams }>('/detect/start', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async stopDetection(): Promise<{ message: string }> {
    return this.request<{ message: string }>('/detect/stop', {
      method: 'POST',
    });
  }

  async saveScreenshot(): Promise<{ message: string; filename: string; filepath: string; timestamp: string }> {
    return this.request<{ message: string; filename: string; filepath: string; timestamp: string }>('/screenshot', {
      method: 'POST',
    });
  }

  // Evaluation
  async startEvaluation(params: EvaluationParams): Promise<{ message: string; model_path: string }> {
    return this.request<{ message: string; model_path: string }>('/evaluate', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // Models
  async getAvailableModels(): Promise<{ models: ModelInfo[] }> {
    return this.request<{ models: ModelInfo[] }>('/models');
  }

  // Datasets
  async getDatasetStatus(): Promise<DatasetStatus> {
    return this.request<DatasetStatus>('/datasets');
  }

  // Model Information
  async getModelInfo(modelPath: string): Promise<ModelDetails> {
    const encodedPath = modelPath.replace(/\//g, '_');
    return this.request<ModelDetails>(`/model-info/${encodedPath}`);
  }

  // Evaluation Results
  async getEvaluationResults(modelPath: string): Promise<EvaluationResults> {
    const encodedPath = modelPath.replace(/\//g, '_');
    return this.request<EvaluationResults>(`/evaluation-results/${encodedPath}`);
  }

  // Training Results
  async getTrainingResults(modelPath: string): Promise<any> {
    const encodedPath = modelPath.replace(/\//g, '_');
    return this.request<any>(`/training-results/${encodedPath}`);
  }
}

export const apiService = new ApiService();
export default apiService;
