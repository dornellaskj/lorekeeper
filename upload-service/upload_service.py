from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import zipfile
from pathlib import Path
from typing import List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lorekeeper File Upload Service", version="1.0.0")

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "104857600"))  # 100MB default
ALLOWED_EXTENSIONS = {
    '.txt', '.md', '.rst', '.log', '.py', '.js', '.html', 
    '.css', '.json', '.xml', '.yaml', '.yml', '.zip'
}

# Ensure upload directory exists
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def upload_page():
    """Serve the upload page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lorekeeper File Upload</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                transition: border-color 0.3s;
            }
            .upload-area:hover {
                border-color: #007bff;
            }
            .upload-area.dragover {
                border-color: #007bff;
                background-color: #f0f8ff;
            }
            input[type="file"] {
                display: none;
            }
            .upload-btn {
                background-color: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
            }
            .upload-btn:hover {
                background-color: #0056b3;
            }
            .file-list {
                margin-top: 20px;
                padding: 0;
                list-style: none;
            }
            .file-item {
                background: #f8f9fa;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .progress-bar {
                width: 100%;
                height: 20px;
                background-color: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-fill {
                height: 100%;
                background-color: #28a745;
                width: 0%;
                transition: width 0.3s;
            }
            .status {
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .status.success {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.error {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .info {
                background-color: #e7f3ff;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .file-count {
                text-align: center;
                margin: 20px 0;
                font-size: 18px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 Lorekeeper File Upload</h1>
            
            <div class="info">
                <strong>Supported file types:</strong> .txt, .md, .rst, .log, .py, .js, .html, .css, .json, .xml, .yaml, .yml, .zip
                <br><strong>Max file size:</strong> 100MB
            </div>

            <div class="upload-area" id="uploadArea">
                <p>📁 Drop files here or click to select</p>
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    Choose Files
                </button>
                <input type="file" id="fileInput" multiple accept=".txt,.md,.rst,.log,.py,.js,.html,.css,.json,.xml,.yaml,.yml,.zip">
            </div>

            <div class="progress-bar" id="progressBar" style="display: none;">
                <div class="progress-fill" id="progressFill"></div>
            </div>

            <div id="status"></div>

            <ul class="file-list" id="fileList"></ul>

            <div class="file-count" id="fileCount">
                <button class="upload-btn" onclick="getFileCount()" style="background-color: #6c757d;">
                    📊 Check Uploaded Files Count
                </button>
            </div>

            <div style="text-align: center; margin-top: 30px;">
                <button class="upload-btn" onclick="triggerDataLoader()" style="background-color: #28a745;">
                    🚀 Start Data Processing
                </button>
            </div>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const fileList = document.getElementById('fileList');
            const status = document.getElementById('status');
            const progressBar = document.getElementById('progressBar');
            const progressFill = document.getElementById('progressFill');

            // Drag and drop handlers
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = Array.from(e.dataTransfer.files);
                handleFiles(files);
            });

            fileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                handleFiles(files);
            });

            async function handleFiles(files) {
                fileList.innerHTML = '';
                showStatus('Uploading files...', 'info');
                progressBar.style.display = 'block';
                
                let completed = 0;
                const total = files.length;

                for (const file of files) {
                    const listItem = document.createElement('li');
                    listItem.className = 'file-item';
                    listItem.innerHTML = `<span>${file.name} (${formatFileSize(file.size)})</span><span>⏳</span>`;
                    fileList.appendChild(listItem);

                    try {
                        await uploadFile(file);
                        listItem.innerHTML = `<span>${file.name} (${formatFileSize(file.size)})</span><span>✅</span>`;
                    } catch (error) {
                        listItem.innerHTML = `<span>${file.name} (${formatFileSize(file.size)})</span><span>❌ ${error.message}</span>`;
                    }

                    completed++;
                    const progress = (completed / total) * 100;
                    progressFill.style.width = `${progress}%`;
                }

                progressBar.style.display = 'none';
                showStatus(`Upload completed! ${completed}/${total} files processed.`, completed === total ? 'success' : 'error');
            }

            async function uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Upload failed');
                }

                return await response.json();
            }

            async function getFileCount() {
                try {
                    const response = await fetch('/files/count');
                    const data = await response.json();
                    document.getElementById('fileCount').innerHTML = `📊 ${data.count} files uploaded`;
                } catch (error) {
                    showStatus('Error getting file count', 'error');
                }
            }

            async function triggerDataLoader() {
                try {
                    showStatus('Starting data processing...', 'info');
                    const response = await fetch('/trigger-loader', { method: 'POST' });
                    const data = await response.json();
                    
                    if (response.ok) {
                        showStatus(data.message, 'success');
                    } else {
                        showStatus(data.detail, 'error');
                    }
                } catch (error) {
                    showStatus('Error starting data processing', 'error');
                }
            }

            function showStatus(message, type) {
                status.innerHTML = `<div class="status ${type}">${message}</div>`;
            }

            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file."""
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Seek back to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Save file
    file_path = Path(UPLOAD_DIR) / file.filename
    
    # Handle duplicate filenames
    counter = 1
    original_path = file_path
    while file_path.exists():
        stem = original_path.stem
        suffix = original_path.suffix
        file_path = original_path.parent / f"{stem}_{counter}{suffix}"
        counter += 1
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # If it's a zip file, extract it
        if file_extension == '.zip':
            extract_zip(file_path)
            os.remove(file_path)  # Remove the zip file after extraction
            
        logger.info(f"File uploaded: {file_path}")
        
        return {
            "filename": file_path.name,
            "size": file_size,
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Error saving file")

def extract_zip(zip_path: Path):
    """Extract zip file contents."""
    extract_dir = zip_path.parent / zip_path.stem
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Security check - prevent directory traversal
            if os.path.isabs(member) or ".." in member:
                continue
                
            # Only extract files with allowed extensions
            if Path(member).suffix.lower() in ALLOWED_EXTENSIONS:
                zip_ref.extract(member, extract_dir)

@app.get("/files/count")
async def get_file_count():
    """Get count of uploaded files."""
    try:
        count = sum(1 for f in Path(UPLOAD_DIR).rglob('*') if f.is_file())
        return {"count": count}
    except Exception as e:
        logger.error(f"Error counting files: {e}")
        raise HTTPException(status_code=500, detail="Error counting files")

@app.get("/files/list")
async def list_files():
    """List all uploaded files."""
    try:
        files = []
        for file_path in Path(UPLOAD_DIR).rglob('*'):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(UPLOAD_DIR)),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {"files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail="Error listing files")

@app.post("/trigger-loader")
async def trigger_data_loader():
    """Trigger the data loader job in Kubernetes."""
    try:
        import subprocess
        
        # Delete existing job if it exists
        subprocess.run([
            "kubectl", "delete", "job", "data-loader", "--ignore-not-found=true"
        ], capture_output=True)
        
        # Create new job
        result = subprocess.run([
            "kubectl", "apply", "-f", "/app/k8s/data-loader-config.yaml"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to apply config: {result.stderr}")
            
        result = subprocess.run([
            "kubectl", "apply", "-f", "/app/k8s/data-loader-job.yaml"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to create job: {result.stderr}")
        
        return {"message": "Data loader job started successfully! Check logs with: kubectl logs -f job/data-loader"}
        
    except Exception as e:
        logger.error(f"Error triggering data loader: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting data loader: {str(e)}")

@app.delete("/files/clear")
async def clear_files():
    """Clear all uploaded files."""
    try:
        shutil.rmtree(UPLOAD_DIR)
        Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        return {"message": "All files cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing files: {e}")
        raise HTTPException(status_code=500, detail="Error clearing files")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)