# VigilAI API Integration

This document describes the integration between the Flask backend API and React frontend for the VigilAI surveillance system.

## Overview

The system consists of:
- **Flask Backend** (`flask_api.py`): Python API server handling model training, detection, and evaluation
- **React Frontend** (`vigilai-ui/`): TypeScript React application with real-time UI updates

## API Endpoints

### System Status
- `GET /api/status` - Get current system status
- `GET /api/logs` - Get recent system logs
- `GET /api/models` - Get available trained models

### Training
- `POST /api/train` - Start model training
  ```json
  {
    "epochs": 50,
    "batch_size": 16,
    "img_size": 640,
    "model_size": "m"
  }
  ```

### Detection
- `POST /api/detect/start` - Start live detection
  ```json
  {
    "camera_index": 0,
    "conf_threshold": 0.5,
    "model_size": "m",
    "model_path": "optional/path/to/model.pt"
  }
  ```
- `POST /api/detect/stop` - Stop live detection

### Evaluation
- `POST /api/evaluate` - Start model evaluation
  ```json
  {
    "model_path": "path/to/model.pt"
  }
  ```

## Frontend Architecture

### API Service (`src/lib/api.ts`)
- Centralized API client using fetch
- TypeScript interfaces for all API requests/responses
- Error handling and response validation

### Custom Hooks (`src/hooks/use-api.ts`)
- `useSystemStatus()` - Real-time system status with polling
- `useLogs()` - Live log updates
- `useAvailableModels()` - Model list management
- `useTraining()` - Training operations
- `useDetection()` - Detection start/stop
- `useEvaluation()` - Evaluation operations
- `useRealTimeStatus()` - Combined status monitoring

### React Query Integration
- Automatic caching and background updates
- Optimistic updates for better UX
- Error handling and retry logic
- Polling for real-time data

## Real-time Features

### Live Status Updates
- System status polling every 1-2 seconds
- Log updates every 2-3 seconds
- Model list refresh every 10 seconds

### Connection Monitoring
- Automatic detection of backend connectivity
- Visual indicators for connection status
- Error messages when backend is unavailable

### State Synchronization
- Frontend state automatically syncs with backend
- Training/detection/evaluation status updates in real-time
- Log messages appear instantly across all pages

## Usage

### Starting the System
1. Run `start_system.bat` to launch both backend and frontend
2. Backend will be available at `http://localhost:5000`
3. Frontend will be available at `http://localhost:5173`

### Manual Start
```bash
# Terminal 1 - Backend
python flask_api.py

# Terminal 2 - Frontend (Vite)
cd vigilai-ui && npm run dev
```

## Error Handling

### Backend Errors
- API errors are caught and displayed as toast notifications
- Connection errors show persistent warning banners
- Retry logic for temporary network issues

### Frontend Errors
- TypeScript compilation errors
- React Query error boundaries
- Graceful degradation when services are unavailable

## Development

### Adding New Endpoints
1. Add endpoint to `flask_api.py`
2. Add interface to `src/lib/api.ts`
3. Create custom hook in `src/hooks/use-api.ts`
4. Update UI components to use the new hook

### Testing
- Backend: Test API endpoints directly with curl/Postman
- Frontend: Use React Query DevTools for debugging
- Integration: Monitor network tab for API calls

## Troubleshooting

### Common Issues
1. **CORS Errors**: Ensure Flask CORS is properly configured
2. **Connection Refused**: Check if Flask server is running on port 5000
3. **Model Not Found**: Verify model paths and file permissions
4. **Training Hangs**: Check GPU memory and dataset paths

### Debug Mode
- Backend: Set `debug=True` in Flask app
- Frontend: Use React Query DevTools
- Logs: Monitor both terminal outputs for errors
