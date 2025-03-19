import { html, css, LitElement } from 'lit';
import * as pdfjs from 'pdfjs-dist';

export class PDFViewer extends LitElement {
    static properties = {
        src: { type: String },
        page: { type: Number },
        zoom: { type: Number },
        annotations: { type: Array }
    };

    constructor() {
        super();
        this.page = 1;
        this.zoom = 1.0;
        this.annotations = [];
        this.totalPages = 0;
        this.pdf = null;
    }

    async firstUpdated() {
        if (this.src) {
            await this.loadPDF();
        }
    }

    async loadPDF() {
        try {
            this.pdf = await pdfjs.getDocument(this.src).promise;
            this.totalPages = this.pdf.numPages;
            await this.renderPage();
        } catch (error) {
            console.error('Error loading PDF:', error);
        }
    }

    async renderPage() {
        if (!this.pdf) return;

        const canvas = this.shadowRoot.querySelector('canvas');
        const context = canvas.getContext('2d');
        
        const page = await this.pdf.getPage(this.page);
        const viewport = page.getViewport({ scale: this.zoom });

        canvas.width = viewport.width;
        canvas.height = viewport.height;

        await page.render({
            canvasContext: context,
            viewport: viewport
        }).promise;

        this.renderAnnotations();
    }

    renderAnnotations() {
        const annotationsLayer = this.shadowRoot.querySelector('.annotations');
        annotationsLayer.innerHTML = '';

        this.annotations
            .filter(a => a.page === this.page)
            .forEach(annotation => {
                const element = document.createElement('div');
                element.className = 'annotation';
                element.style.left = `${annotation.x * this.zoom}px`;
                element.style.top = `${annotation.y * this.zoom}px`;
                element.textContent = annotation.text;
                annotationsLayer.appendChild(element);
            });
    }

    render() {
        return html`
            <div class="pdf-viewer">
                <div class="toolbar">
                    <button 
                        ?disabled="${this.page === 1}"
                        @click="${() => this.page--}">
                        Previous
                    </button>
                    <span>${this.page} / ${this.totalPages}</span>
                    <button 
                        ?disabled="${this.page === this.totalPages}"
                        @click="${() => this.page++}">
                        Next
                    </button>
                </div>

                <div class="content">
                    <canvas></canvas>
                    <div class="annotations"></div>
                </div>
            </div>
        `;
    }

    static styles = css`
        :host {
            display: block;
        }

        .pdf-viewer {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .toolbar {
            display: flex;
            gap: 1rem;
            padding: 0.5rem;
            background: var(--toolbar-bg, #f5f5f5);
            align-items: center;
            justify-content: center;
        }

        .content {
            position: relative;
            flex: 1;
            overflow: auto;
        }

        canvas {
            display: block;
        }

        .annotations {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
        }

        .annotation {
            position: absolute;
            background: rgba(255, 255, 0, 0.3);
            padding: 4px;
            border-radius: 2px;
        }
    `;
}

customElements.define('pdf-viewer', PDFViewer); 