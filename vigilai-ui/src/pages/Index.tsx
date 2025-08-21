import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/status-badge";
import { ControlCard } from "@/components/ui/control-card";
import { Activity, Play, Square, Camera, AlertCircle, RefreshCw } from "lucide-react";
import { useRealTimeStatus, useLogs } from "@/hooks/use-api";

const Dashboard = () => {
  const navigate = useNavigate();
  const { 
    isConnected, 
    isTraining, 
    isDetecting, 
    isEvaluating, 
    isIdle,
    error: connectionError 
  } = useRealTimeStatus();
  const { data: logsData, isLoading: logsLoading, error: logsError, refetch: refetchLogs } = useLogs(10, 3000); // Get last 10 logs, refresh every 3 seconds

  // Determine system status based on real-time data
  const getSystemStatus = () => {
    if (isTraining) return "training";
    if (isDetecting) return "detecting";
    if (isEvaluating) return "evaluating";
    return "idle";
  };

  const handleStartTraining = () => {
    navigate("/training#params");
  };

  const handleStartDetection = () => {
    navigate("/detection#camera");
  };

  const handleStartEvaluation = () => {
    navigate("/evaluation#setup");
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* System Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Activity className="h-6 w-6" />
          <h1 className="font-display text-2xl tracking-tight">System Status</h1>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-xs font-mono">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <StatusBadge status={getSystemStatus()} />
        </div>
      </div>

      {/* Connection Error */}
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

      {/* Control Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Training Control */}
        <ControlCard title="Training Module">
          <div className="space-y-3">
            <p className="text-sm font-mono text-muted-foreground">
              Train YOLOv11 model with custom dataset
            </p>
            <Button 
              onClick={handleStartTraining}
              disabled={!isIdle}
              className="w-full font-mono"
            >
              <Play className="h-4 w-4 mr-2" />
              Start Training
            </Button>
          </div>
        </ControlCard>

        {/* Detection Control */}
        <ControlCard title="Live Detection">
          <div className="space-y-3">
            <p className="text-sm font-mono text-muted-foreground">
              Real-time object detection from camera feed
            </p>
            <Button 
              onClick={handleStartDetection}
              disabled={!isIdle}
              className="w-full font-mono"
            >
              <Camera className="h-4 w-4 mr-2" />
              Start Detection
            </Button>
          </div>
        </ControlCard>

        {/* Evaluation Control */}
        <ControlCard title="Model Evaluation">
          <div className="space-y-3">
            <p className="text-sm font-mono text-muted-foreground">
              Evaluate model performance metrics
            </p>
            <Button 
              onClick={handleStartEvaluation}
              disabled={!isIdle}
              className="w-full font-mono"
            >
              <Activity className="h-4 w-4 mr-2" />
              Start Evaluation
            </Button>
          </div>
        </ControlCard>
      </div>

      {/* Activity Log */}
      <ControlCard 
        title="System Activity Log"
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
              {isConnected ? 'No logs available' : 'Cannot fetch logs - disconnected'}
            </div>
          )}
        </div>
      </ControlCard>
    </div>
  );
};

export default Dashboard;
