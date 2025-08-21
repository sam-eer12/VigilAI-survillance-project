import { useState, useEffect, useCallback, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiService, { 
  SystemStatus, 
  LogEntry, 
  TrainingParams, 
  DetectionParams, 
  EvaluationParams,
  ModelInfo 
} from '@/lib/api';

// Query keys
export const queryKeys = {
  status: ['status'] as const,
  logs: ['logs'] as const,
  models: ['models'] as const,
  datasets: ['datasets'] as const,
  modelInfo: (path: string) => ['model-info', path] as const,
  evaluationResults: (path: string) => ['evaluation-results', path] as const,
};

// Custom hook for system status with polling
export const useSystemStatus = (pollingInterval = 2000) => {
  return useQuery({
    queryKey: queryKeys.status,
    queryFn: () => apiService.getStatus(),
    refetchInterval: pollingInterval,
    refetchIntervalInBackground: true,
    staleTime: 1000,
  });
};

// Custom hook for logs with polling
export const useLogs = (limit = 50, pollingInterval = 3000) => {
  return useQuery({
    queryKey: queryKeys.logs,
    queryFn: () => apiService.getLogs(limit),
    refetchInterval: pollingInterval,
    refetchIntervalInBackground: true,
    staleTime: 2000,
    retry: 3,
    retryDelay: 1000,
  });
};

// Custom hook for available models
export const useAvailableModels = () => {
  return useQuery({
    queryKey: queryKeys.models,
    queryFn: () => apiService.getAvailableModels(),
    refetchInterval: 10000, // Refresh every 10 seconds
    staleTime: 5000,
  });
};

// Custom hook for dataset status
export const useDatasetStatus = () => {
  return useQuery({
    queryKey: queryKeys.datasets,
    queryFn: () => apiService.getDatasetStatus(),
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 10000,
  });
};

// Custom hook for model information
export const useModelInfo = (modelPath: string) => {
  return useQuery({
    queryKey: queryKeys.modelInfo(modelPath),
    queryFn: () => apiService.getModelInfo(modelPath),
    enabled: !!modelPath,
    staleTime: 60000, // 1 minute
  });
};

// Custom hook for evaluation results
export const useEvaluationResults = (modelPath: string) => {
  return useQuery({
    queryKey: queryKeys.evaluationResults(modelPath),
    queryFn: () => apiService.getEvaluationResults(modelPath),
    enabled: !!modelPath,
    staleTime: 300000, // 5 minutes
  });
};

// Custom hook for training results
export const useTrainingResults = (modelPath: string) => {
  return useQuery({
    queryKey: ['training-results', modelPath] as const,
    queryFn: () => apiService.getTrainingResults(modelPath),
    enabled: !!modelPath,
    staleTime: 300000, // 5 minutes
  });
};

// Custom hook for training
export const useTraining = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (params: TrainingParams) => apiService.startTraining(params),
    onSuccess: () => {
      // Invalidate and refetch status and logs
      queryClient.invalidateQueries({ queryKey: queryKeys.status });
      queryClient.invalidateQueries({ queryKey: queryKeys.logs });
    },
  });
};

// Custom hook for detection
export const useDetection = () => {
  const queryClient = useQueryClient();
  
  const startDetection = useMutation({
    mutationFn: (params: DetectionParams) => apiService.startDetection(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.status });
      queryClient.invalidateQueries({ queryKey: queryKeys.logs });
    },
  });

  const stopDetection = useMutation({
    mutationFn: () => apiService.stopDetection(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.status });
      queryClient.invalidateQueries({ queryKey: queryKeys.logs });
    },
  });

  const saveScreenshot = useMutation({
    mutationFn: () => apiService.saveScreenshot(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.logs });
    },
  });

  return { startDetection, stopDetection, saveScreenshot };
};

// Custom hook for evaluation
export const useEvaluation = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (params: EvaluationParams) => apiService.startEvaluation(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.status });
      queryClient.invalidateQueries({ queryKey: queryKeys.logs });
    },
  });
};

// Custom hook for real-time status monitoring
export const useRealTimeStatus = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const statusRef = useRef<SystemStatus | null>(null);

  const { data: status, error, isLoading } = useSystemStatus(1000);

  useEffect(() => {
    if (status) {
      statusRef.current = status;
      setLastUpdate(new Date());
      setIsConnected(true);
    }
  }, [status]);

  useEffect(() => {
    if (error) {
      setIsConnected(false);
    }
  }, [error]);

  const isTraining = status?.status === 'Training';
  const isDetecting = status?.detection_running || status?.status === 'Detecting';
  const isEvaluating = status?.status === 'Evaluating';
  const isIdle = status?.status === 'Idle';

  return {
    status: statusRef.current,
    isConnected,
    lastUpdate,
    isLoading,
    error,
    isTraining,
    isDetecting,
    isEvaluating,
    isIdle,
  };
};
