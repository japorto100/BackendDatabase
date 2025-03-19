import { html, css, LitElement } from 'lit';

class ResultsContainer extends LitElement {
    static properties = {
        results: { type: Array },    // Ergebnisse von allen Providern
        activeProvider: { type: String },
        loading: { type: Boolean },
        error: { type: String },
        view: { type: String }       // 'list', 'grid', oder 'detailed'
    };

    constructor() {
        super();
        this.results = [];
        this.loading = false;
        this.view = 'list';
    }

    render() {
        return html`
            <div class="results-container">
                <!-- Provider Tabs -->
                <div class="provider-tabs">
                    <button class="${this.activeProvider === 'all' ? 'active' : ''}"
                            @click="${() => this.activeProvider = 'all'}">
                        All Results
                    </button>
                    <button class="${this.activeProvider === 'apollo' ? 'active' : ''}"
                            @click="${() => this.activeProvider = 'apollo'}">
                        Apollo.io
                    </button>
                    <button class="${this.activeProvider === 'eu_data' ? 'active' : ''}"
                            @click="${() => this.activeProvider = 'eu_data'}">
                        EU Data
                    </button>
                </div>

                <!-- View Controls -->
                <div class="view-controls">
                    <button @click="${() => this.view = 'list'}">📝 List</button>
                    <button @click="${() => this.view = 'grid'}">📊 Grid</button>
                    <button @click="${() => this.view = 'detailed'}">📋 Detailed</button>
                </div>

                <!-- Results Display -->
                ${this.loading ? 
                    html`<div class="loading">Loading results...</div>` :
                    this.renderResults()}
            </div>
        `;
    }

    renderResults() {
        // Filtere nach aktivem Provider
        const filteredResults = this.activeProvider === 'all' 
            ? this.results 
            : this.results.filter(r => r.provider === this.activeProvider);

        // Wähle das passende Render-Template
        switch(this.view) {
            case 'grid':
                return this.renderGridView(filteredResults);
            case 'detailed':
                return this.renderDetailedView(filteredResults);
            default:
                return this.renderListView(filteredResults);
        }
    }

    renderListView(results) {
        return html`
            <div class="results-list">
                ${results.map(result => html`
                    <div class="result-item">
                        <h3>${result.title}</h3>
                        <p>${result.description}</p>
                        <div class="meta">
                            <span>${result.provider}</span>
                            <span>${result.date}</span>
                        </div>
                    </div>
                `)}
            </div>
        `;
    }

    renderGridView(results) {
        return html`
            <div class="results-grid">
                ${results.map(result => html`
                    <div class="grid-item">
                        <div class="preview">
                            ${result.image 
                                ? html`<img src="${result.image}" alt="${result.title}">` 
                                : html`<div class="placeholder">${result.provider[0]}</div>`
                            }
                        </div>
                        <div class="content">
                            <h3>${result.title}</h3>
                            <p class="truncate">${result.description}</p>
                            <div class="meta">
                                <span class="provider">${result.provider}</span>
                                <span class="date">${result.date}</span>
                            </div>
                        </div>
                        <div class="actions">
                            <button @click="${() => this.openDetail(result)}">
                                View Details
                            </button>
                            <button @click="${() => this.saveResult(result)}">
                                Save
                            </button>
                        </div>
                    </div>
                `)}
            </div>
        `;
    }

    renderDetailedView(results) {
        return html`
            <div class="detailed-view">
                ${results.map(result => html`
                    <div class="detailed-item">
                        <div class="header">
                            <h2>${result.title}</h2>
                            <div class="badges">
                                <span class="provider-badge ${result.provider}">
                                    ${result.provider}
                                </span>
                                <span class="type-badge">
                                    ${result.type || 'Unknown'}
                                </span>
                            </div>
                        </div>

                        <div class="content-wrapper">
                            <!-- Left Column: Main Content -->
                            <div class="main-content">
                                ${result.image ? html`
                                    <div class="image-container">
                                        <img src="${result.image}" alt="${result.title}">
                                    </div>
                                ` : ''}
                                
                                <div class="description">
                                    ${result.description}
                                </div>

                                ${result.highlights ? html`
                                    <div class="highlights">
                                        <h4>Highlights</h4>
                                        <ul>
                                            ${result.highlights.map(highlight => html`
                                                <li>${highlight}</li>
                                            `)}
                                        </ul>
                                    </div>
                                ` : ''}
                            </div>

                            <!-- Right Column: Metadata & Actions -->
                            <div class="sidebar">
                                <div class="metadata">
                                    <div class="meta-item">
                                        <span class="label">Source:</span>
                                        <span class="value">${result.source}</span>
                                    </div>
                                    <div class="meta-item">
                                        <span class="label">Date:</span>
                                        <span class="value">${result.date}</span>
                                    </div>
                                    ${result.url ? html`
                                        <div class="meta-item">
                                            <span class="label">URL:</span>
                                            <a href="${result.url}" target="_blank">Open Link</a>
                                        </div>
                                    ` : ''}
                                </div>

                                <div class="actions">
                                    <button class="primary" @click="${() => this.saveResult(result)}">
                                        Save Result
                                    </button>
                                    <button @click="${() => this.shareResult(result)}">
                                        Share
                                    </button>
                                    ${result.type === 'document' ? html`
                                        <button @click="${() => this.openInViewer(result)}">
                                            Open in Viewer
                                        </button>
                                    ` : ''}
                                </div>
                            </div>
                        </div>
                    </div>
                `)}
            </div>
        `;
    }

    // Event Handlers
    openDetail(result) {
        this.dispatchEvent(new CustomEvent('open-detail', { detail: result }));
    }

    saveResult(result) {
        this.dispatchEvent(new CustomEvent('save-result', { detail: result }));
    }

    shareResult(result) {
        this.dispatchEvent(new CustomEvent('share-result', { detail: result }));
    }

    openInViewer(result) {
        this.dispatchEvent(new CustomEvent('open-in-viewer', { detail: result }));
    }

    static styles = css`
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            padding: 1rem;
        }

        .provider-tabs {
            display: flex;
            gap: 0.5rem;
        }

        .result-item {
            padding: 1rem;
            border: 1px solid var(--border-color, #ddd);
            border-radius: 4px;
            margin-bottom: 0.5rem;
        }

        .meta {
            display: flex;
            gap: 1rem;
            font-size: 0.9em;
            color: #666;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
            padding: 1rem;
        }

        .grid-item {
            display: flex;
            flex-direction: column;
            border: 1px solid var(--border-color, #ddd);
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .grid-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .preview {
            height: 150px;
            background: var(--preview-bg, #f5f5f5);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .detailed-view {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .detailed-item {
            border: 1px solid var(--border-color, #ddd);
            border-radius: 8px;
            overflow: hidden;
        }

        .content-wrapper {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 1rem;
            padding: 1rem;
        }

        .sidebar {
            border-left: 1px solid var(--border-color, #ddd);
            padding-left: 1rem;
        }

        .metadata {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .meta-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-color, #ddd);
        }

        .badges {
            display: flex;
            gap: 0.5rem;
        }

        .provider-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }

        .provider-badge.apollo {
            background: #e3f2fd;
            color: #1976d2;
        }

        .provider-badge.eu_data {
            background: #e8f5e9;
            color: #2e7d32;
        }

        @media (max-width: 768px) {
            .content-wrapper {
                grid-template-columns: 1fr;
            }

            .sidebar {
                border-left: none;
                border-top: 1px solid var(--border-color, #ddd);
                padding-left: 0;
                padding-top: 1rem;
            }
        }
    `;
}

customElements.define('results-container', ResultsContainer); 