import { html, css, LitElement } from 'lit';

// Diese Imports werden nicht benötigt, da wir die Custom Elements über ihre Tags verwenden
// und die Komponenten sich selbst registrieren
// import './pdf-viewer.js';
// import './code-viewer.js';
// import './image-viewer.js';
// import './spreadsheet-viewer.js';

class DocumentViewer extends LitElement {
    static properties = {
        document: { type: Object },
        loading: { type: Boolean },
        error: { type: String },
        viewMode: { type: String }, // 'single', 'split', 'grid'
        zoom: { type: Number },
        currentPage: { type: Number },
        annotations: { type: Array },
        searchQuery: { type: String }
    };

    constructor() {
        super();
        this.loading = false;
        this.error = null;
        this.viewMode = 'single';
        this.zoom = 1.0;
        this.currentPage = 1;
        this.annotations = [];
        this.searchQuery = '';
    }

    getViewer(document) {
        const viewers = {
            'application/pdf': () => html`
                <pdf-viewer
                    .src="${document.url}"
                    .page="${this.currentPage}"
                    .zoom="${this.zoom}"
                    .annotations="${this.annotations}"
                    @pageChange="${this.handlePageChange}"
                    @annotationAdd="${this.handleAnnotationAdd}"
                ></pdf-viewer>
            `,
            'image': () => html`
                <image-viewer
                    .src="${document.url}"
                    .zoom="${this.zoom}"
                    .annotations="${this.annotations}"
                    @annotationAdd="${this.handleAnnotationAdd}"
                ></image-viewer>
            `,
            'text/code': () => html`
                <code-viewer
                    .content="${document.content}"
                    .language="${document.language}"
                    .highlights="${this.annotations}"
                    .lineNumbers="${true}"
                ></code-viewer>
            `,
            'spreadsheet': () => html`
                <spreadsheet-viewer
                    .data="${document.content}"
                    .activeSheet="${document.activeSheet}"
                    .filters="${document.filters}"
                    @sortChange="${this.handleSortChange}"
                ></spreadsheet-viewer>
            `
        };

        const viewer = viewers[document.type];
        return viewer ? viewer() : html`
            <div class="error">Unsupported document type: ${document.type}</div>
        `;
    }

    render() {
        if (this.loading) {
            return html`<div class="loading">Loading document...</div>`;
        }

        if (this.error) {
            return html`<div class="error">${this.error}</div>`;
        }

        return html`
            <div class="document-viewer ${this.viewMode}">
                <!-- Toolbar -->
                <div class="toolbar">
                    <div class="view-controls">
                        <button @click="${() => this.viewMode = 'single'}">Single</button>
                        <button @click="${() => this.viewMode = 'split'}">Split</button>
                        <button @click="${() => this.viewMode = 'grid'}">Grid</button>
                    </div>
                    
                    <div class="zoom-controls">
                        <button @click="${() => this.zoom -= 0.1}">-</button>
                        <span>${Math.round(this.zoom * 100)}%</span>
                        <button @click="${() => this.zoom += 0.1}">+</button>
                    </div>

                    <div class="search-box">
                        <input 
                            type="text"
                            .value="${this.searchQuery}"
                            @input="${e => this.searchQuery = e.target.value}"
                            placeholder="Search in document..."
                        >
                    </div>
                </div>

                <!-- Main Viewer -->
                <div class="viewer-container">
                    ${this.getViewer(this.document)}
                </div>

                <!-- Annotations Panel -->
                <div class="annotations-panel">
                    <h3>Annotations</h3>
                    ${this.annotations.map(annotation => html`
                        <div class="annotation">
                            <span>${annotation.text}</span>
                            <button @click="${() => this.removeAnnotation(annotation.id)}">
                                Delete
                            </button>
                        </div>
                    `)}
                </div>
            </div>
        `;
    }

    static styles = css`
        :host {
            display: block;
            font-family: Arial, sans-serif;
            color: #333;
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .toolbar {
    static styles = css`
}   