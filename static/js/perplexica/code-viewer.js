import { html, css, LitElement } from 'lit';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import Prism from 'prismjs';

export class CodeViewer extends LitElement {
    static properties = {
        content: { type: String },
        language: { type: String },
        highlights: { type: Array },
        lineNumbers: { type: Boolean }
    };

    constructor() {
        super();
        this.content = '';
        this.language = 'javascript';
        this.highlights = [];
        this.lineNumbers = true;
    }

    highlightCode() {
        const highlighted = Prism.highlight(
            this.content,
            Prism.languages[this.language],
            this.language
        );
        return this.lineNumbers 
            ? Prism.plugins.lineNumbers.wrap(highlighted)
            : highlighted;
    }

    render() {
        return html`
            <div class="code-viewer ${this.language}">
                <div class="header">
                    <span class="language">${this.language}</span>
                    <button @click="${() => navigator.clipboard.writeText(this.content)}">
                        Copy
                    </button>
                </div>
                
                <pre class="code-block"><code>${unsafeHTML(this.highlightCode())}</code></pre>
                
                ${this.highlights.map(highlight => html`
                    <div class="highlight" 
                        style="top: ${highlight.line * 20}px">
                        ${highlight.note}
                    </div>
                `)}
            </div>
        `;
    }

    static styles = css`
        :host {
            display: block;
        }

        .code-viewer {
            background: var(--code-bg, #1e1e1e);
            color: var(--code-color, #d4d4d4);
            border-radius: 4px;
            overflow: hidden;
        }

        .header {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem;
            background: rgba(255, 255, 255, 0.1);
        }

        .code-block {
            margin: 0;
            padding: 1rem;
            overflow-x: auto;
        }

        .highlight {
            position: absolute;
            left: 0;
            right: 0;
            background: rgba(255, 255, 0, 0.1);
            pointer-events: none;
        }
    `;
}

customElements.define('code-viewer', CodeViewer); 