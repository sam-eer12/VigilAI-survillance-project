import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ControlCard } from "@/components/ui/control-card";
import { StatusBadge } from "@/components/ui/status-badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BarChart3, Play, FileText, AlertCircle, RefreshCw, Info, TrendingUp } from "lucide-react";
import { 
  useEvaluation, 
  useLogs, 
  useAvailableModels, 
  useRealTimeStatus, 
  useModelInfo, 
  useEvaluationResults,
  useTrainingResults
} from "@/hooks/use-api";
import { useToast } from "@/hooks/use-toast";

const Evaluation = () => {
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [evaluationType, setEvaluationType] = useState<"standard" | "combined">("standard");

  const { toast } = useToast();
  const { mutate: startEvaluation, isPending: isEvaluating } = useEvaluation();
  const { data: logsData, isLoading: logsLoading, error: logsError, refetch: refetchLogs } = useLogs(20, 2000);
  const { data: modelsData } = useAvailableModels();
  const { isEvaluating: isEvaluatingStatus, isConnected, error: connectionError } = useRealTimeStatus();
  
  // Get model information and evaluation results
  const { data: modelInfo, isLoading: modelInfoLoading } = useModelInfo(selectedModel);
  const { data: evaluationResults, isLoading: resultsLoading, refetch: refetchResults } = useEvaluationResults(selectedModel);
  const { data: trainingResults, isLoading: trainingLoading, refetch: refetchTrainingResults } = useTrainingResults(selectedModel);

  // Auto-select the latest model
  useEffect(() => {
    if (modelsData?.models && modelsData.models.length > 0 && !selectedModel) {
      setSelectedModel(modelsData.models[0].path);
    }
  }, [modelsData, selectedModel]);

  // Auto-set evaluation type based on model type
  useEffect(() => {
    if (modelInfo?.can_evaluate_combined) {
      setEvaluationType("combined");
    } else {
      setEvaluationType("standard");
    }
  }, [modelInfo]);

  const handleRunEvaluation = () => {
    if (!selectedModel) {
      toast({
        title: "Error",
        description: "Please select a model to evaluate",
        variant: "destructive",
      });
      return;
    }

    try {
      startEvaluation({ 
        model_path: selectedModel,
        evaluation_type: evaluationType
      });
      
      toast({
        title: "Evaluation Started",
        description: `${evaluationType === 'combined' ? 'Combined' : 'Standard'} evaluation has begun. Check logs for progress.`,
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to start evaluation",
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    if (window.location.hash) {
      const id = window.location.hash.replace('#', '');
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <BarChart3 className="h-6 w-6" />
          <h1 className="font-display text-2xl tracking-tight">Model Evaluation</h1>
        </div>
        <StatusBadge status={isEvaluating ? "evaluating" : "idle"} />
      </div>

      {/* Connection Status */}
      {connectionError && (
        <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-destructive" />
          <div>
            <p className="font-medium text-destructive">Connection Error</p>
            <p className="text-sm text-muted-foreground">
              Unable to connect to VigilAI backend. Please ensure the Flask server is running.
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Evaluation Setup */}
        <ControlCard title="Evaluation Setup">
          <div className="space-y-4" id="setup">
            <div className="space-y-2">
              <Label className="font-mono text-sm">Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isEvaluating}>
                <SelectTrigger className="font-mono">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {modelsData?.models?.map((model) => (
                    <SelectItem key={model.path} value={model.path}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {modelsData?.models?.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  No trained models found. Please train a model first.
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label className="font-mono text-sm">Evaluation Type</Label>
              <Select 
                value={evaluationType} 
                onValueChange={(value: "standard" | "combined") => setEvaluationType(value)} 
                disabled={isEvaluating || !modelInfo?.can_evaluate_combined}
              >
                <SelectTrigger className="font-mono">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="standard">Standard (HackByte Dataset)</SelectItem>
                  <SelectItem value="combined" disabled={!modelInfo?.can_evaluate_combined}>
                    Combined (All Datasets)
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {evaluationType === 'combined' 
                  ? 'Evaluates on all three datasets: HackByte + FireSmoke + Violence Detection'
                  : 'Evaluates on HackByte dataset only: FireExtinguisher, ToolBox, OxygenTank'
                }
              </p>
            </div>

            <Button 
              onClick={handleRunEvaluation}
              disabled={isEvaluating || !isConnected || !selectedModel}
              className="w-full font-mono"
            >
              <Play className="h-4 w-4 mr-2" />
              {isEvaluating ? "Evaluating..." : "Run Evaluation"}
            </Button>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="font-mono text-sm">Evaluation Log</Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => refetchLogs()}
                  disabled={logsLoading}
                  className="h-6 w-6 p-0"
                >
                  <RefreshCw className={`h-3 w-3 ${logsLoading ? 'animate-spin' : ''}`} />
                </Button>
              </div>
              <div className="log-container max-h-[300px] overflow-y-auto">
                {logsLoading ? (
                  <div className="py-1 text-xs text-muted-foreground">
                    Loading logs...
                  </div>
                ) : logsError ? (
                  <div className="py-1 text-xs text-destructive">
                    Error loading logs: {logsError.message}
                  </div>
                ) : logsData?.logs?.length > 0 ? (
                  logsData.logs.map((log, index) => (
                    <div key={index} className="py-1 text-xs border-b border-border/50 last:border-b-0">
                      <span className="text-muted-foreground">[{log.timestamp}]</span>
                      <span className={`ml-2 ${
                        log.level === 'ERROR' ? 'text-destructive' : 
                        log.level === 'WARNING' ? 'text-yellow-600' : 
                        'text-foreground'
                      }`}>
                        {log.message}
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="py-1 text-xs text-muted-foreground">
                    No logs available
                  </div>
                )}
              </div>
            </div>
          </div>
        </ControlCard>

        {/* Training Results */}
        <ControlCard title="Training Results">
          {trainingLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
              <p className="font-mono text-sm text-muted-foreground">
                Loading training results...
              </p>
            </div>
          ) : trainingResults && !trainingResults.error ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-green-600" />
                  <span className="font-mono text-sm font-semibold">Final Epoch Results</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => refetchTrainingResults()}
                  className="h-6 w-6 p-0"
                >
                  <RefreshCw className="h-3 w-3" />
                </Button>
              </div>
              
              {/* Key Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <div className="font-mono text-xs text-muted-foreground">mAP@0.5</div>
                  <div className="font-mono text-2xl font-bold text-green-600">
                    {(trainingResults.mAP50 * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="font-mono text-xs text-muted-foreground">mAP@0.5:0.95</div>
                  <div className="font-mono text-2xl font-bold text-blue-600">
                    {(trainingResults.mAP50_95 * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border">
                <div className="space-y-1">
                  <div className="font-mono text-xs text-muted-foreground">Precision</div>
                  <div className="font-mono text-lg font-bold">
                    {(trainingResults.precision * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="font-mono text-xs text-muted-foreground">Recall</div>
                  <div className="font-mono text-lg font-bold">
                    {(trainingResults.recall * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Training Info */}
              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border">
                <div className="space-y-1">
                  <div className="font-mono text-xs text-muted-foreground">Total Epochs</div>
                  <div className="font-mono text-sm font-bold">{trainingResults.total_epochs}</div>
                </div>
                <div className="space-y-1">
                  <div className="font-mono text-xs text-muted-foreground">Training Time</div>
                  <div className="font-mono text-sm font-bold">{Math.round(trainingResults.training_time / 60)} min</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <FileText className="h-12 w-12 mx-auto mb-2 text-muted-foreground" />
              <p className="font-mono text-sm text-muted-foreground">
                {trainingResults?.error ? 'No training results found' : 'Select a model to view training results'}
              </p>
              {trainingResults?.message && (
                <p className="font-mono text-xs text-muted-foreground mt-1">
                  {trainingResults.message}
                </p>
              )}
            </div>
          )}
        </ControlCard>
      </div>

      {/* Loss Information Panel */}
      {trainingResults && !trainingResults.error && (
        <ControlCard title="Loss & Training Details">
          <div className="space-y-6">
            {/* Losses */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <AlertCircle className="h-4 w-4 text-orange-600" />
                <span className="font-mono text-sm font-semibold">Final Loss Values</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {/* Training Losses */}
                <div className="space-y-2">
                  <div className="font-mono text-xs text-muted-foreground font-bold">Training Losses</div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="font-mono text-xs">Box Loss:</span>
                      <span className="font-mono text-xs font-bold text-red-600">{trainingResults.train_box_loss.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono text-xs">Class Loss:</span>
                      <span className="font-mono text-xs font-bold text-red-600">{trainingResults.train_cls_loss.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono text-xs">DFL Loss:</span>
                      <span className="font-mono text-xs font-bold text-red-600">{trainingResults.train_dfl_loss.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
                
                {/* Validation Losses */}
                <div className="space-y-2">
                  <div className="font-mono text-xs text-muted-foreground font-bold">Validation Losses</div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="font-mono text-xs">Box Loss:</span>
                      <span className="font-mono text-xs font-bold text-orange-600">{trainingResults.val_box_loss.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono text-xs">Class Loss:</span>
                      <span className="font-mono text-xs font-bold text-orange-600">{trainingResults.val_cls_loss.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono text-xs">DFL Loss:</span>
                      <span className="font-mono text-xs font-bold text-orange-600">{trainingResults.val_dfl_loss.toFixed(4)}</span>
                    </div>
                  </div>
                </div>

                {/* Learning Rates */}
                <div className="space-y-2">
                  <div className="font-mono text-xs text-muted-foreground font-bold">Learning Rates</div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="font-mono text-xs">LR PG0:</span>
                      <span className="font-mono text-xs font-bold text-blue-600">{trainingResults.lr_pg0.toExponential(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono text-xs">LR PG1:</span>
                      <span className="font-mono text-xs font-bold text-blue-600">{trainingResults.lr_pg1.toExponential(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-mono text-xs">LR PG2:</span>
                      <span className="font-mono text-xs font-bold text-blue-600">{trainingResults.lr_pg2.toExponential(2)}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Additional Metrics Info */}
            <div className="pt-4 border-t border-border">
              <div className="flex items-center gap-2 mb-3">
                <Info className="h-4 w-4 text-blue-600" />
                <span className="font-mono text-sm font-semibold">Loss Explanation</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div className="space-y-1">
                  <div className="font-mono font-bold">Box Loss</div>
                  <div className="text-muted-foreground">Measures how well the model predicts bounding box coordinates</div>
                </div>
                <div className="space-y-1">
                  <div className="font-mono font-bold">Class Loss</div>
                  <div className="text-muted-foreground">Measures how well the model classifies objects into correct categories</div>
                </div>
                <div className="space-y-1">
                  <div className="font-mono font-bold">DFL Loss</div>
                  <div className="text-muted-foreground">Distribution Focal Loss for more accurate bounding box regression</div>
                </div>
              </div>
            </div>
          </div>
        </ControlCard>
      )}

      {/* Model Information */}
      <ControlCard title="Model Information">
        {modelInfoLoading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
            <p className="font-mono text-sm text-muted-foreground">
              Loading model information...
            </p>
          </div>
        ) : modelInfo ? (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm font-mono">
              <div className="space-y-1">
                <div className="font-bold">Model Type</div>
                <div className="text-muted-foreground">{modelInfo.model_type}</div>
              </div>
              <div className="space-y-1">
                <div className="font-bold">Parameters</div>
                <div className="text-muted-foreground">{modelInfo.parameters}</div>
              </div>
              <div className="space-y-1">
                <div className="font-bold">Model Size</div>
                <div className="text-muted-foreground">{modelInfo.model_size_mb} MB</div>
              </div>
              <div className="space-y-1">
                <div className="font-bold">Inference Speed</div>
                <div className="text-muted-foreground">{modelInfo.inference_speed}</div>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4 text-blue-600" />
                <span className="font-bold text-sm">Classes</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {modelInfo.classes.map((className, index) => (
                  <div key={index} className="text-xs bg-muted/50 rounded px-2 py-1 text-center">
                    {className}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <Info className="h-12 w-12 mx-auto mb-2 text-muted-foreground" />
            <p className="font-mono text-sm text-muted-foreground">
              Select a model to view information
            </p>
          </div>
        )}
      </ControlCard>
    </div>
  );
};

export default Evaluation;