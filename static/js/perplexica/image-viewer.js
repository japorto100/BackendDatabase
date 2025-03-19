import { html, css, LitElement } from 'lit';

export class ImageViewer extends LitElement {
    static properties = {
        src: { type: String },
        zoom: { type: Number },
        annotations: { type: Array },
        rotation: { type: Number }
    };

    constructor() {
        super();
        this.zoom = 1.0;
        this.annotations = [];
        this.rotation = 0;
    }

    render() {
        return html`
            <div class="image-viewer" 
                style="transform: scale(${this.zoom}) rotate(${this.rotation}deg)">
                <img src="${this.src}" alt="Document Image" />
                
                ${this.annotations.map(annotation => html`
                    <div class="annotation" 
                        style="left: ${annotation.x}px; top: ${annotation.y}px">
                        ${annotation.text}
                    </div>
                `)}
            </div>

            <div class="controls">
                <button @click="${() => this.rotation = (this.rotation + 90) % 360}">
                    Rotate
                </button>
            </div>
        `;
    }

    static styles = css`
        :host {
            display: block;
            position: relative;
            overflow: hidden;
        }

        .image-viewer {
            position: relative;
            transition: transform 0.3s ease;
            transform-origin: center center;
        }

        img {
            max-width: 100%;
            height: auto;
        }

        .annotation {
            position: absolute;
            background: rgba(255, 255, 0, 0.3);
            padding: 4px;
            border-radius: 2px;
            pointer-events: none;
        }

        .controls {
            position: absolute;
            bottom: 1rem;
            right: 1rem;
            display: flex;
            gap: 0.5rem;
        }
    `;
}

customElements.define('image-viewer', ImageViewer); 