# Upload Files to Kubernetes PVC and Run Data Loader
# Run this from the lorekeeper root directory

param(
    [string]$LocalPath = ".\DataInput",
    [switch]$SkipUpload,
    [switch]$SkipDataLoader
)

Write-Host "🚀 Lorekeeper Data Upload and Processing Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if kubectl is available
if (!(Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Host "❌ kubectl not found. Please install kubectl first." -ForegroundColor Red
    exit 1
}

# Check if local files exist
if (!(Test-Path $LocalPath)) {
    Write-Host "❌ Local path '$LocalPath' not found." -ForegroundColor Red
    exit 1
}

$fileCount = (Get-ChildItem $LocalPath -Recurse -File).Count
Write-Host "📁 Found $fileCount files in '$LocalPath'" -ForegroundColor Green

if (!$SkipUpload) {
    Write-Host "`n🔧 Step 1: Deploying PVC..." -ForegroundColor Yellow
    kubectl apply -f dataloader/k8s/data-input-pvc.yaml
    
    Write-Host "🔧 Step 2: Creating upload pod..." -ForegroundColor Yellow
    kubectl run file-upload-pod --image=busybox --restart=Never --overrides='{
      "spec": {
        "containers": [{
          "name": "upload",
          "image": "busybox", 
          "command": ["sleep", "3600"],
          "volumeMounts": [{
            "name": "data-storage",
            "mountPath": "/data"
          }]
        }],
        "volumes": [{
          "name": "data-storage",
          "persistentVolumeClaim": {
            "claimName": "data-input-pvc"
          }
        }]
      }
    }'
    
    Write-Host "⏳ Waiting for pod to be ready..." -ForegroundColor Yellow
    kubectl wait --for=condition=Ready pod/file-upload-pod --timeout=60s
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Pod failed to start" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "📤 Step 3: Uploading files from '$LocalPath'..." -ForegroundColor Yellow
    kubectl cp "$LocalPath/." file-upload-pod:/data/
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ File upload failed" -ForegroundColor Red
        kubectl delete pod file-upload-pod --force
        exit 1
    }
    
    Write-Host "✅ Verifying uploaded files..." -ForegroundColor Green
    $uploadedFiles = kubectl exec file-upload-pod -- find /data -type f | Measure-Object -Line
    Write-Host "📊 Uploaded $($uploadedFiles.Lines) files successfully" -ForegroundColor Green
    
    Write-Host "🧹 Cleaning up upload pod..." -ForegroundColor Yellow
    kubectl delete pod file-upload-pod --force
}

if (!$SkipDataLoader) {
    Write-Host "`n🔄 Step 4: Running data loader..." -ForegroundColor Yellow
    kubectl apply -f dataloader/k8s/data-loader-config.yaml
    kubectl apply -f dataloader/k8s/data-loader-job.yaml
    
    Write-Host "📊 Monitoring data loader progress..." -ForegroundColor Green
    Write-Host "Use Ctrl+C to stop monitoring (job will continue running)" -ForegroundColor Gray
    
    try {
        kubectl logs -f job/data-loader
    } catch {
        Write-Host "Log monitoring stopped" -ForegroundColor Gray
    }
    
    Write-Host "`n✅ Data loader job submitted!" -ForegroundColor Green
    Write-Host "Monitor with: kubectl logs -f job/data-loader" -ForegroundColor Cyan
    Write-Host "Check status: kubectl get jobs" -ForegroundColor Cyan
}

Write-Host "`n🎉 Process completed!" -ForegroundColor Green