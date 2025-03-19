import { html, css, LitElement } from 'lit';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';

export class FilePreview extends LitElement {
    static properties = {
        filePath: { type: String },
        fileName: { type: String },
        fileContent: { type: String },
        fileType: { type: String },
        isExpanded: { type: Boolean },
        isLoading: { type: Boolean },
        error: { type: String },
        highlights: { type: Array },
        showConfidence: { type: Boolean }
    };

    constructor() {
        super();
        this.filePath = '';
        this.fileName = '';
        this.fileContent = '';
        this.fileType = '';
        this.isExpanded = false;
        this.isLoading = false;
        this.error = null;
        this.highlights = [];
        this.showConfidence = false;
    }

    connectedCallback() {
        super.connectedCallback();
        if (this.filePath) {
            this.loadFileContent();
        }
    }

    updated(changedProperties) {
        if (changedProperties.has('filePath') && this.filePath) {
            this.loadFileContent();
        }
    }

    async loadFileContent() {
        if (!this.filePath) return;
        
        this.isLoading = true;
        this.error = null;
        
        try {
            const response = await fetch(`/api/files/preview?path=${encodeURIComponent(this.filePath)}`);
            if (!response.ok) {
                throw new Error(`Failed to load file: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.fileContent = data.content;
            this.fileName = data.name || this.fileName || this.filePath.split('/').pop();
            this.fileType = data.type || this.fileType || this.getFileTypeFromPath(this.filePath);
            
        } catch (error) {
            console.error('Error loading file:', error);
            this.error = error.message;
        } finally {
            this.isLoading = false;
        }
    }

    getFileTypeFromPath(path) {
        const extension = path.split('.').pop().toLowerCase();
        const typeMap = {
            'md': 'markdown',
            'js': 'javascript',
            'py': 'python',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'txt': 'text'
        };
        return typeMap[extension] || 'text';
    }

    toggleExpand() {
        this.isExpanded = !this.isExpanded;
    }

    renderContent() {
        if (!this.fileContent) return '';
        
        if (!this.highlights || this.highlights.length === 0) {
            return html`<pre class="code-block ${this.fileType}"><code>${this.fileContent}</code></pre>`;
        }
        
        // Create highlighted content
        let highlightedContent = '';
        let lastEnd = 0;
        
        // Sort highlights by position
        const sortedHighlights = [...this.highlights].sort((a, b) => a.start - b.start);
        
        for (const highlight of sortedHighlights) {
            // Add text before highlight
            highlightedContent += this.escapeHtml(this.fileContent.substring(lastEnd, highlight.start));
            
            // Add highlighted text with optional confidence styling
            const confidenceClass = this.showConfidence ? 
                ` confidence-${Math.floor(highlight.confidence * 10)}` : '';
            
            highlightedContent += `<mark class="evidence-highlight${confidenceClass}" 
                                      data-id="${highlight.id || ''}" 
                                      title="${highlight.confidence ? `Confidence: ${Math.round(highlight.confidence * 100)}%` : ''}">
                                  ${this.escapeHtml(this.fileContent.substring(highlight.start, highlight.end))}
                              </mark>`;
            
            lastEnd = highlight.end;
        }
        
        // Add remaining text
        highlightedContent += this.escapeHtml(this.fileContent.substring(lastEnd));
        
        return html`<pre class="code-block ${this.fileType}"><code>${unsafeHTML(highlightedContent)}</code></pre>`;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    render() {
        return html`
            <div class="file-preview ${this.isExpanded ? 'expanded' : 'collapsed'}">
                <div class="file-header" @click=${this.toggleExpand}>
                    <span class="toggle-icon">${this.isExpanded ? '▼' : '▶'}</span>
                    <span class="file-name">@${this.fileName}</span>
                    <span class="file-path">${this.filePath}</span>
                </div>
                
                ${this.isExpanded ? html`
                    <div class="file-content">
                        ${this.isLoading ? html`
                            <div class="loading">Loading file content...</div>
                        ` : this.error ? html`
                            <div class="error">${this.error}</div>
                        ` : this.renderContent()}
                    </div>
                ` : ''}
            </div>
        `;
    }

    static styles = css`
        .file-preview {
            border: 1px solid var(--border-color, #ddd);
            border-radius: 4px;
            margin-bottom: 8px;
            background-color: var(--bg-color, #f8f8f8);
            overflow: hidden;
        }
        
        .file-header {
            display: flex;
            align-items: center;
            padding: 8px 12px;
            cursor: pointer;
            background-color: var(--header-bg, #2d2d2d);
            color: var(--header-color, #e0e0e0);
            font-family: monospace;
        }
        
        .file-header:hover {
            background-color: var(--header-hover-bg, #3d3d3d);
        }
        
        .toggle-icon {
            margin-right: 8px;
            font-size: 12px;
        }
        
        .file-name {
            font-weight: bold;
            margin-right: 12px;
            color: var(--file-name-color, #f1c40f);
        }
        
        .file-path {
            font-size: 12px;
            color: var(--path-color, #aaa);
            margin-left: auto;
        }
        
        .file-content {
            max-height: 500px;
            overflow-y: auto;
            background-color: var(--content-bg, #1e1e1e);
            border-top: 1px solid var(--border-color, #333);
            padding: 0;
        }
        
        .code-block {
            margin: 0;
            padding: 12px;
            overflow-x: auto;
            font-family: monospace;
            font-size: 14px;
            line-height: 1.5;
            color: var(--code-color, #e0e0e0);
        }
        
        .loading, .error {
            padding: 12px;
            text-align: center;
        }
        
        .error {
            color: var(--error-color, #e74c3c);
        }
        
        /* Evidence highlight styling */
        .evidence-highlight {
            background-color: rgba(255, 204, 0, 0.3);
            border-radius: 2px;
            padding: 2px 0;
        }
        
        /* Confidence levels (if enabled) */
        .confidence-10 { background-color: rgba(0, 255, 0, 0.4); }
        .confidence-9 { background-color: rgba(40, 255, 0, 0.4); }
        .confidence-8 { background-color: rgba(80, 255, 0, 0.4); }
        .confidence-7 { background-color: rgba(120, 255, 0, 0.4); }
        .confidence-6 { background-color: rgba(160, 255, 0, 0.4); }
        .confidence-5 { background-color: rgba(200, 255, 0, 0.4); }
        .confidence-4 { background-color: rgba(255, 240, 0, 0.4); }
        .confidence-3 { background-color: rgba(255, 200, 0, 0.4); }
        .confidence-2 { background-color: rgba(255, 160, 0, 0.4); }
        .confidence-1 { background-color: rgba(255, 120, 0, 0.4); }
        .confidence-0 { background-color: rgba(255, 80, 0, 0.4); }
        
        /* Syntax highlighting classes */
        .markdown {
            color: var(--md-color, #e0e0e0);
        }
        
        .javascript {
            color: var(--js-color, #f8c555);
        }
        
        .python {
            color: var(--py-color, #4da6ff);
        }
    `;
}

customElements.define('file-preview', FilePreview); 