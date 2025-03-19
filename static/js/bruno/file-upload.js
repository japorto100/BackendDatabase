import { html, css, LitElement } from 'lit';
import FilesAPI from '../api/files-api.js';

/**
 * File Upload Component
 * 
 * A Bruno component for uploading and managing files
 */
class FileUpload extends LitElement {
    static styles = css`
        :host {
            display: block;
            font-family: var(--bruno-font-family, sans-serif);
        }
        
        .file-upload {
            border: 2px dashed var(--bruno-border-color, #ccc);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .file-upload:hover, .file-upload.dragover {
            border-color: var(--bruno-primary-color, #4a90e2);
            background-color: var(--bruno-hover-bg, #f5f8ff);
        }
        
        .file-list {
            margin-top: 20px;
        }
        
        .file-item {
            display: flex;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid var(--bruno-border-color, #eee);
        }
        
        .file-name {
            flex: 1;
            margin-right: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .file-type {
            margin-right: 10px;
            color: var(--bruno-text-secondary, #666);
        }
        
        .file-actions {
            display: flex;
            gap: 5px;
        }
        
        button {
            background-color: var(--bruno-button-bg, #f5f5f5);
            border: 1px solid var(--bruno-border-color, #ccc);
            border-radius: 4px;
            padding: 5px 10px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        button:hover {
            background-color: var(--bruno-button-hover-bg, #e5e5e5);
        }
        
        button.primary {
            background-color: var(--bruno-primary-color, #4a90e2);
            color: white;
            border-color: var(--bruno-primary-color, #4a90e2);
        }
        
        button.primary:hover {
            background-color: var(--bruno-primary-dark, #3a80d2);
        }
        
        button.danger {
            background-color: var(--bruno-danger-color, #e74c3c);
            color: white;
            border-color: var(--bruno-danger-color, #e74c3c);
        }
        
        button.danger:hover {
            background-color: var(--bruno-danger-dark, #c0392b);
        }
        
        .progress {
            height: 5px;
            width: 100%;
            background-color: var(--bruno-progress-bg, #f5f5f5);
            border-radius: 3px;
            margin-top: 10px;
        }
        
        .progress-bar {
            height: 100%;
            background-color: var(--bruno-primary-color, #4a90e2);
            border-radius: 3px;
            transition: width 0.3s ease;
        }
    `;
    
    static properties = {
        files: { type: Array },
        uploading: { type: Boolean },
        progress: { type: Number },
        error: { type: String }
    };
    
    constructor() {
        super();
        this.files = [];
        this.uploading = false;
        this.progress = 0;
        this.error = '';
    }
    
    connectedCallback() {
        super.connectedCallback();
        this.loadFiles();
    }
    
    async loadFiles() {
        try {
            this.files = await FilesAPI.getFiles();
        } catch (error) {
            this.error = 'Failed to load files';
            console.error(error);
        }
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.currentTarget.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.uploadFile(file);
        }
    }
    
    async uploadFile(file) {
        this.uploading = true;
        this.progress = 0;
        this.error = '';
        
        // Simulate progress
        const progressInterval = setInterval(() => {
            if (this.progress < 90) {
                this.progress += 10;
                this.requestUpdate();
            }
        }, 300);
        
        try {
            const uploadedFile = await FilesAPI.uploadFile(file);
            this.progress = 100;
            
            // Add the new file to the list
            this.files = [uploadedFile, ...this.files];
            
            // Dispatch event
            this.dispatchEvent(new CustomEvent('file-uploaded', {
                detail: { file: uploadedFile }
            }));
        } catch (error) {
            this.error = error.message || 'Failed to upload file';
            this.progress = 0;
        } finally {
            clearInterval(progressInterval);
            setTimeout(() => {
                this.uploading = false;
                this.progress = 0;
            }, 1000);
        }
    }
    
    async deleteFile(fileId) {
        try {
            await FilesAPI.deleteFile(fileId);
            
            // Remove the file from the list
            this.files = this.files.filter(file => file.id !== fileId);
            
            // Dispatch event
            this.dispatchEvent(new CustomEvent('file-deleted', {
                detail: { fileId }
            }));
        } catch (error) {
            this.error = 'Failed to delete file';
            console.error(error);
        }
    }
    
    async processFile(fileId) {
        try {
            const result = await FilesAPI.processFile(fileId);
            
            // Update the file in the list
            this.files = this.files.map(file => {
                if (file.id === fileId) {
                    return { ...file, processed: true, processing_results: result.results };
                }
                return file;
            });
            
            // Dispatch event
            this.dispatchEvent(new CustomEvent('file-processed', {
                detail: { fileId, results: result.results }
            }));
        } catch (error) {
            this.error = 'Failed to process file';
            console.error(error);
        }
    }
    
    render() {
        return html`
            <div class="file-upload"
                @dragover="${this.handleDragOver}"
                @dragleave="${this.handleDragLeave}"
                @drop="${this.handleDrop}">
                <p>Drag and drop a file here, or</p>
                <input type="file" id="file-input" @change="${this.handleFileSelect}" style="display: none;">
                <button class="primary" @click="${() => this.shadowRoot.querySelector('#file-input').click()}">
                    Select File
                </button>
                
                ${this.uploading ? html`
                    <div class="progress">
                        <div class="progress-bar" style="width: ${this.progress}%"></div>
                    </div>
                ` : ''}
                
                ${this.error ? html`<p style="color: red;">${this.error}</p>` : ''}
            </div>
            
            <div class="file-list">
                <h3>Your Files</h3>
                ${this.files.length === 0 ? html`<p>No files uploaded yet</p>` : ''}
                ${this.files.map(file => html`
                    <div class="file-item">
                        <div class="file-name">${file.original_filename}</div>
                        <div class="file-type">${file.file_type}</div>
                        <div class="file-actions">
                            <button @click="${() => window.open(`/files/download/${file.id}/`, '_blank')}">
                                Download
                            </button>
                            ${file.file_type === 'document' && !file.processed ? html`
                                <button @click="${() => this.processFile(file.id)}">
                                    Process
                                </button>
                            ` : ''}
                            <button class="danger" @click="${() => this.deleteFile(file.id)}">
                                Delete
                            </button>
                        </div>
                    </div>
                `)}
            </div>
        `;
    }
}

customElements.define('bruno-file-upload', FileUpload); 