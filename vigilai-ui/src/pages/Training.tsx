import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ControlCard } from "@/components/ui/control-card";
import { StatusBadge } from "@/components/ui/status-badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Brain, Play, Square, AlertCircle, RefreshCw } from "lucide-react";
import { useTraining, useLogs, useRealTimeStatus, useDatasetStatus } from "@/hooks/use-api";
import { useToast } from "@/hooks/use-toast";

const Training = () => {
  const [epochs, setEpochs] = useState("50");
  const [batchSize, setBatchSize] = useState("16");
  const [imageSize, setImageSize] = useState("640");
  const [modelSize, setModelSize] = useState("m");
  const [trainingType, setTrainingType] = useState<"standard" | "combined">("standard");

  const { toast } = useToast();
  const { mutate: startTraining, isPending: isTrainingStarting } = useTraining();
  const { data: logsData, isLoading: logsLoading, error: logsError, refetch: refetchLogs } = useLogs(30, 2000);
  const { isTraining, isConnected, error: connectionError } = useRealTimeStatus();
  const { data: datasetStatus } = useDatasetStatus();

  const handleStartTraining = () => {
    try {
      startTraining({
        epochs: parseInt(epochs),
        batch_size: parseInt(batchSize),
        img_size: parseInt(imageSize),
        model_size: modelSize,
        training_type: trainingType,
      });
      
      toast({
        title: "Training Started",
        description: `${trainingType === 'combined' ? 'Combined' : 'Standard'} model training has begun. Check logs for progress.`,
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to start training",
        variant: "destructive",
      });
    }
  };

  const handleStopTraining = () => {
    // Note: The Flask API doesn't currently support stopping training
    // This would need to be implemented on the backend
    toast({
      title: "Training Stop",
      description: "Training stop functionality coming soon",
    });
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
          <Brain className="h-6 w-6" />
          <h1 className="font-display text-2xl tracking-tight">Model Training</h1>
        </div>
        <StatusBadge status={isTraining ? "training" : "idle"} />
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
        {/* Training Parameters */}
        <ControlCard title="Training Parameters">
          <div className="space-y-4" id="params">
            <div className="space-y-2">
              <Label className="font-mono text-sm">Training Type</Label>
              <Select value={trainingType} onValueChange={(value: "standard" | "combined") => setTrainingType(value)} disabled={isTraining}>
                <SelectTrigger className="font-mono">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="standard">Standard (HackByte Dataset)</SelectItem>
                  <SelectItem value="combined" disabled={!datasetStatus?.can_train_combined}>
                    Combined (All Datasets)
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {trainingType === 'combined' 
                  ? 'Trains on all three datasets: HackByte + FireSmoke + Violence Detection'
                  : 'Trains on HackByte dataset only: FireExtinguisher, ToolBox, OxygenTank'
                }
              </p>
              {trainingType === 'combined' && !datasetStatus?.can_train_combined && (
                <p className="text-xs text-destructive">
                  ⚠️ Combined training requires at least 2 datasets to be available
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label className="font-mono text-sm">Model Size</Label>
              <Select value={modelSize} onValueChange={setModelSize} disabled={isTraining}>
                <SelectTrigger className="font-mono">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="n">Nano (fastest)</SelectItem>
                  <SelectItem value="s">Small</SelectItem>
                  <SelectItem value="m">Medium (recommended)</SelectItem>
                  <SelectItem value="l">Large</SelectItem>
                  <SelectItem value="x">Extra Large (best)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="font-mono text-sm">Epochs</Label>
              <Input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(e.target.value)}
                disabled={isTraining}
                min="1"
                max="1000"
                className="font-mono"
              />
            </div>

            <div className="space-y-2">
              <Label className="font-mono text-sm">Batch Size</Label>
              <Input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(e.target.value)}
                disabled={isTraining}
                min="1"
                max="64"
                className="font-mono"
              />
            </div>

            <div className="space-y-2">
              <Label className="font-mono text-sm">Image Size (pixels)</Label>
              <Input
                type="number"
                value={imageSize}
                onChange={(e) => setImageSize(e.target.value)}
                disabled={isTraining}
                min="320"
                max="1280"
                step="32"
                className="font-mono"
              />
            </div>

            <div className="pt-4">
              {!isTraining ? (
                <Button 
                  onClick={handleStartTraining} 
                  className="w-full font-mono"
                  disabled={
                    isTrainingStarting || 
                    !isConnected || 
                    (trainingType === 'standard' && !datasetStatus?.datasets?.['HackByte_Dataset']) ||
                    (trainingType === 'combined' && !datasetStatus?.can_train_combined)
                  }
                >
                  <Play className="h-4 w-4 mr-2" />
                  {isTrainingStarting ? "Starting..." : "Start Training"}
                </Button>
              ) : (
                <Button 
                  onClick={handleStopTraining} 
                  variant="destructive" 
                  className="w-full font-mono"
                >
                  <Square className="h-4 w-4 mr-2" />
                  Stop Training
                </Button>
              )}
            </div>
          </div>
        </ControlCard>

        {/* Training Progress */}
        <ControlCard title="Training Progress">
          <div className="space-y-4">
            {isTraining && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm font-mono">
                  <span>Status</span>
                  <span className="text-green-600">Training Active</span>
                </div>
                <Progress value={100} className="w-full" />
                <p className="text-xs text-muted-foreground">
                  Check logs below for detailed progress
                </p>
              </div>
            )}

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="font-mono text-sm">Training Logs</Label>
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
      </div>

      {/* Dataset Information */}
      <ControlCard title="Dataset Information">
        {trainingType === 'combined' ? (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm font-mono">
              <div className="space-y-1">
                <div className="font-bold">Total Classes</div>
                <div className="text-muted-foreground">8 classes</div>
              </div>
              <div className="space-y-1">
                <div className="font-bold">Available Datasets</div>
                <div className="text-muted-foreground">
                  {datasetStatus?.available_count || 0}/{datasetStatus?.total_count || 0}
                </div>
              </div>
              <div className="space-y-1">
                <div className="font-bold">Training Time</div>
                <div className="text-muted-foreground">2-4 hours</div>
              </div>
            </div>
            
            {/* Dataset Status */}
            <div className="space-y-2">
              <div className="font-bold text-sm">Dataset Status:</div>
              <div className="grid grid-cols-1 gap-1 text-xs">
                {datasetStatus?.datasets && Object.entries(datasetStatus.datasets).map(([name, available]) => (
                  <div key={name} className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${available ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className={available ? 'text-foreground' : 'text-muted-foreground'}>
                      {name.replace('.v1i.yolov9', '').replace('_', ' ')}
                    </span>
                    <span className="text-muted-foreground">
                      {available ? '(Available)' : '(Missing)'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="font-bold text-sm">Classes:</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="space-y-1">
                  <div className="font-semibold">HackByte Dataset:</div>
                  <div className="text-muted-foreground">• FireExtinguisher</div>
                  <div className="text-muted-foreground">• ToolBox</div>
                  <div className="text-muted-foreground">• OxygenTank</div>
                </div>
                <div className="space-y-1">
                  <div className="font-semibold">FireSmoke Dataset:</div>
                  <div className="text-muted-foreground">• fire</div>
                  <div className="text-muted-foreground">• smoke</div>
                  <div className="text-muted-foreground">• other</div>
                </div>
                <div className="space-y-1">
                  <div className="font-semibold">Violence Dataset:</div>
                  <div className="text-muted-foreground">• violence</div>
                  <div className="text-muted-foreground">• non-violence</div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm font-mono">
              <div className="space-y-1">
                <div className="font-bold">Training Images</div>
                <div className="text-muted-foreground">1,248 images</div>
              </div>
              <div className="space-y-1">
                <div className="font-bold">Validation Images</div>
                <div className="text-muted-foreground">312 images</div>
              </div>
              <div className="space-y-1">
                <div className="font-bold">Classes</div>
                <div className="text-muted-foreground">3 classes</div>
              </div>
            </div>
            
            {/* Dataset Status for Standard Training */}
            <div className="space-y-2">
              <div className="font-bold text-sm">Required Dataset:</div>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${datasetStatus?.datasets?.['HackByte_Dataset'] ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className={datasetStatus?.datasets?.['HackByte_Dataset'] ? 'text-foreground' : 'text-muted-foreground'}>
                  HackByte Dataset
                </span>
                <span className="text-muted-foreground">
                  {datasetStatus?.datasets?.['HackByte_Dataset'] ? '(Available)' : '(Missing)'}
                </span>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="font-bold text-sm">Classes:</div>
              <div className="text-muted-foreground">• FireExtinguisher</div>
              <div className="text-muted-foreground">• ToolBox</div>
              <div className="text-muted-foreground">• OxygenTank</div>
            </div>
          </div>
        )}
      </ControlCard>
    </div>
  );
};

export default Training;