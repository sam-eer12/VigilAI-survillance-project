import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ControlCard } from "@/components/ui/control-card";
import { StatusBadge } from "@/components/ui/status-badge";
import { Slider } from "@/components/ui/slider";
import { Camera, Play, Square, Download, AlertCircle, RefreshCw, ImageIcon } from "lucide-react";
import { useDetection, useLogs, useAvailableModels, useRealTimeStatus } from "@/hooks/use-api";
import apiService from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const Detection = () => {
  const [confidence, setConfidence] = useState([0.5]);
  const [modelSize, setModelSize] = useState("m");
  const [cameraIndex, setCameraIndex] = useState("0");
  const [selectedModel, setSelectedModel] = useState<string>("");

  const { toast } = useToast();
  const { startDetection, stopDetection, saveScreenshot } = useDetection();
  const { data: logsData, isLoading: logsLoading, error: logsError, refetch: refetchLogs } = useLogs(20, 2000);
  const { data: modelsData } = useAvailableModels();
  const { isDetecting, isConnected, error: connectionError } = useRealTimeStatus();

  // Auto-select the latest model
  useEffect(() => {
    if (modelsData?.models && modelsData.models.length > 0 && !selectedModel) {
      setSelectedModel(modelsData.models[0].path);
    }
  }, [modelsData, selectedModel]);

  const handleStartDetection = async () => {
    try {
      await startDetection.mutateAsync({
        camera_index: parseInt(cameraIndex),
        conf_threshold: confidence[0],
        model_size: modelSize,
        model_path: selectedModel || undefined,
      });
      
      toast({
        title: "Detection Started",
        description: "Live detection is now running",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to start detection",
        variant: "destructive",
      });
    }
  };

  const handleStopDetection = async () => {
    try {
      await stopDetection.mutateAsync();
      
      toast({
        title: "Detection Stopped",
        description: "Live detection has been stopped",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to stop detection",
        variant: "destructive",
      });
    }
  };

  const handleSaveScreenshot = async () => {
    try {
      const result = await saveScreenshot.mutateAsync();
      
      toast({
        title: "Screenshot Saved Successfully",
        description: `Screenshot saved as ${result.filename} in screenshots folder`,
      });
    } catch (error) {
      toast({
        title: "Screenshot Failed",
        description: error instanceof Error ? error.message : "Failed to save screenshot",
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
          <Camera className="h-6 w-6" />
          <h1 className="font-display text-2xl tracking-tight">Live Detection</h1>
        </div>
        <StatusBadge status={isDetecting ? "detecting" : "idle"} />
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

      
        {/* Camera Stream */}
        <div className="space-y-6">
          <ControlCard title="Camera Feed">
            <div className="camera-stream flex items-center justify-center min-h-[400px] bg-muted/20 rounded-lg overflow-hidden">
              {isDetecting ? (
                <img
                  src={`${apiService.getBaseUrl().replace(/\/$/, '')}/stream`}
                  alt="Live detection stream"
                  className="w-full h-full object-contain bg-black"
                />
              ) : (
                <div className="text-center">
                  <Camera className="h-12 w-12 mx-auto mb-2 text-muted-foreground" />
                  <p className="font-mono text-sm text-muted-foreground">
                    Camera feed will appear here
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">
                    Start detection to begin monitoring and enable screenshots
                  </p>
                </div>
              )}
            </div>

            <div className="flex gap-3 pt-4" id="camera">
              {!isDetecting ? (
                <Button 
                  onClick={handleStartDetection} 
                  className="font-mono"
                  disabled={startDetection.isPending}
                >
                  <Play className="h-4 w-4 mr-2" />
                  {startDetection.isPending ? "Starting..." : "Start Detection"}
                </Button>
              ) : (
                <Button 
                  onClick={handleStopDetection} 
                  variant="destructive" 
                  className="font-mono"
                  disabled={stopDetection.isPending}
                >
                  <Square className="h-4 w-4 mr-2" />
                  {stopDetection.isPending ? "Stopping..." : "Stop Detection"}
                </Button>
              )}
              
              <Button 
                onClick={handleSaveScreenshot}
                variant="secondary"
                disabled={!isDetecting || saveScreenshot.isPending}
                className="font-mono"
              >
                {saveScreenshot.isPending ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <ImageIcon className="h-4 w-4 mr-2" />
                    Screenshot
                  </>
                )}
              </Button>
            </div>
          </ControlCard>
        

        {/* Control Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ControlCard title="Detection Parameters">
            <div className="space-y-4">
              {/* Model Selection */}
              <div className="space-y-2">
                <Label className="font-mono text-sm">Model</Label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
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

              {/* Confidence Threshold */}
              <div className="space-y-2">
                <Label className="font-mono text-sm">
                  Confidence Threshold: {confidence[0].toFixed(2)}
                </Label>
                <Slider
                  value={confidence}
                  onValueChange={setConfidence}
                  min={0}
                  max={1}
                  step={0.01}
                  className="w-full"
                />
              </div>

              {/* Model Size */}
              <div className="space-y-2">
                <Label className="font-mono text-sm">Model Size</Label>
                <Select value={modelSize} onValueChange={setModelSize}>
                  <SelectTrigger className="font-mono">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="n">Nano (fastest)</SelectItem>
                    <SelectItem value="s">Small</SelectItem>
                    <SelectItem value="m">Medium</SelectItem>
                    <SelectItem value="l">Large</SelectItem>
                    <SelectItem value="x">Extra Large (best)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Camera Index */}
              <div className="space-y-2">
                <Label className="font-mono text-sm">Camera Index</Label>
                <Input
                  type="number"
                  value={cameraIndex}
                  onChange={(e) => setCameraIndex(e.target.value)}
                  min="0"
                  max="10"
                  className="font-mono"
                />
              </div>
            </div>
          </ControlCard>

          <ControlCard 
            title="System Logs"
            action={
              <Button
                variant="ghost"
                size="sm"
                onClick={() => refetchLogs()}
                disabled={logsLoading}
                className="h-6 w-6 p-0"
              >
                <RefreshCw className={`h-3 w-3 ${logsLoading ? 'animate-spin' : ''}`} />
              </Button>
            }
          >
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
          </ControlCard>

          <ControlCard title="Connection Status">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-xs font-mono">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div className="text-xs text-muted-foreground">
                Camera: {cameraIndex} | Confidence: {confidence[0].toFixed(2)}
              </div>
            </div>
          </ControlCard>
        </div>
      </div>
    </div>
  );
};

export default Detection;