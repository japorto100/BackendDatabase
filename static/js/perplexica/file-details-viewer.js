import { html, css, LitElement } from 'lit';

export class FileDetailsViewer extends LitElement {
    static properties = {
        file: { type: Object },
        expanded: { type: Object },
        loading: { type: Boolean },
        error: { type: String }
    };

    constructor() {
        super();
        this.file = null;
        this.expanded = {};
        this.loading = false;
        this.error = null;
    }

    toggleSection(sectionId) {
        this.expanded = {
            ...this.expanded,
            [sectionId]: !this.expanded[sectionId]
        };
    }

    renderFileSection(section, index) {
        const sectionId = `section-${index}`;
        const isExpanded = this.expanded[sectionId] || false;
        
        return html`
            <div class="file-section">
                <div class="section-header" @click=${() => this.toggleSection(sectionId)}>
                    <span class="toggle-icon">${isExpanded ? '▼' : '▶'}</span>
                    <span class="file-name">${section.name}</span>
                    <span class="file-path">${section.path}</span>
                    <span class="line-range">${section.startLine} - ${section.endLine}</span>
                </div>
                ${isExpanded ? html`
                    <div class="section-content">
                        <code-viewer 
                            .content=${section.content}
                            .language=${section.language}
                            .highlights=${section.highlights || []}
                            .lineNumbers=${true}
                        ></code-viewer>
                    </div>
                ` : ''}
            </div>
        `;
    }

    render() {
        if (this.loading) {
            return html`<div class="loading">Loading file details...</div>`;
        }

        if (this.error) {
            return html`<div class="error">${this.error}</div>`;
        }

        if (!this.file) {
            return html`<div class="empty">No file selected</div>`;
        }

        return html`
            <div class="file-details-viewer">
                <div class="file-header">
                    <h3>${this.file.name}</h3>
                    <div class="file-meta">
                        <span class="file-type">${this.file.type}</span>
                        <span class="file-size">${this.formatFileSize(this.file.size)}</span>
                    </div>
                </div>
                
                <div class="file-sections">
                    ${this.file.sections.map((section, index) => 
                        this.renderFileSection(section, index)
                    )}
                </div>
            </div>
        `;
    }

    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }

    static styles = css`
        :host {
            display: block;
            font-family: var(--font-family, 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif);
            color: var(--text-color, #333);
            background-color: var(--bg-color, #f9f9f9);
            border-radius: 4px;
            overflow: hidden;
        }

        .file-details-viewer {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .file-header {
            padding: 1rem;
            background-color: var(--header-bg, #e9e9e9);
            border-bottom: 1px solid var(--border-color, #ddd);
        }

        .file-header h3 {
            margin: 0 0 0.5rem 0;
            font-size: 1.2rem;
        }

        .file-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
            color: var(--meta-color, #666);
        }

        .file-sections {
            flex: 1;
            overflow-y: auto;
        }

        .file-section {
            border-bottom: 1px solid var(--border-color, #ddd);
        }

        .section-header {
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            cursor: pointer;
            background-color: var(--section-header-bg, #f0f0f0);
            transition: background-color 0.2s;
        }

        .section-header:hover {
            background-color: var(--section-header-hover-bg, #e0e0e0);
        }

        .toggle-icon {
            margin-right: 0.5rem;
            font-size: 0.8rem;
            color: var(--icon-color, #666);
        }

        .file-name {
            font-weight: 500;
            margin-right: 1rem;
        }

        .file-path {
            color: var(--path-color, #666);
            font-size: 0.9rem;
            margin-right: auto;
        }

        .line-range {
            color: var(--line-range-color, #666);
            font-size: 0.9rem;
            padding: 0.2rem 0.5rem;
            background-color: var(--line-range-bg, #e0e0e0);
            border-radius: 4px;
        }

        .section-content {
            border-top: 1px solid var(--border-color, #ddd);
            background-color: var(--content-bg, #fff);
        }

        .loading, .error, .empty {
            padding: 2rem;
            text-align: center;
        }

        .error {
            color: var(--error-color, #dc3545);
        }
    `;
}

customElements.define('file-details-viewer', FileDetailsViewer); 